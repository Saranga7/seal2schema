from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .data import build_train_val_datasets, validate_paired_metadata
from .loss import mask_losses, skeleton_losses, supervised_reconstruction_loss, weighted_l1
from .metrics import reconstruction_metrics
from .models.dino_decoder import make_dino_decoder
from .models.diffusion import make_diffusion
from .models.dpt_decoder import make_dpt_decoder
from .models.pix2pix import make_pix2pix
from .models.resnet_unet import make_resnet_unet
from .models.sam_decoder import make_sam_decoder
from .models.sd import train_stable_controlnet
from .splits import create_splits
from .utils import get_device, save_json, seed_everything


def _wandb_cfg(cfg: DictConfig) -> DictConfig | None:
    return cfg.train.get("wandb") if "wandb" in cfg.train else None


def _wandb_enabled(cfg: DictConfig) -> bool:
    wandb_cfg = _wandb_cfg(cfg)
    return bool(wandb_cfg is not None and wandb_cfg.get("enabled", False))


def _start_wandb(cfg: DictConfig):
    if not _wandb_enabled(cfg):
        return None
    import wandb

    wandb_cfg = _wandb_cfg(cfg)
    run_name = wandb_cfg.get("name") or f"{cfg.mode}_{cfg.model.name}_{Path(cfg.output_dir).name}"
    return wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        name=run_name,
        group=wandb_cfg.get("group"),
        mode=wandb_cfg.get("mode", "online"),
        tags=list(wandb_cfg.get("tags", [])),
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(Path(cfg.output_dir)),
    )


def _wandb_log(run: Any, payload: dict[str, Any], step: int | None = None) -> None:
    if run is not None:
        run.log(payload, step=step)


def _wandb_image(path: Path, caption: str):
    import wandb

    return wandb.Image(str(path), caption=caption)


def _loader(dataset, cfg: DictConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=shuffle,
        num_workers=int(cfg.data.num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def _expected_input_channels(input_mode: str) -> int:
    if input_mode == "seal_mask":
        return 4
    if input_mode == "mask_only":
        return 1
    if input_mode == "seal_only":
        return 3
    raise ValueError(f"Unknown data.input_mode: {input_mode}")


def _validate_input_channels(cfg: DictConfig) -> None:
    expected = _expected_input_channels(str(cfg.data.input_mode))
    actual = int(cfg.model.in_channels)
    if expected != actual:
        raise ValueError(
            f"data.input_mode={cfg.data.input_mode} produces {expected} channel(s), "
            f"but model.in_channels={actual}. Set model.in_channels={expected}."
        )


def _is_better_metric(current: float, best: float, mode: str) -> bool:
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ValueError(f"Unknown selection_mode: {mode}")


@torch.no_grad()
def _validate_generator(generator: nn.Module, loader: DataLoader, device: torch.device, cfg: DictConfig, epoch: int) -> tuple[dict, Path | None]:
    generator.eval()
    totals: dict[str, float] = {}
    n = 0
    sample_path = None
    for batch_idx, batch in enumerate(loader):
        if cfg.train.max_eval_batches is not None and batch_idx >= int(cfg.train.max_eval_batches):
            break
        seal = batch["input"].to(device)
        target = batch["schema"].to(device)
        pred = generator(seal).clamp(-1, 1)
        metrics = reconstruction_metrics(pred, target, cfg.evaluate.threshold)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        n += 1
        if batch_idx == 0:
            sample_dir = Path(cfg.output_dir) / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            seal_vis = (batch.get("seal_rgb", seal[:, :3])[:4, :1].detach().cpu() + 1) / 2
            mask_vis = (batch.get("mask", seal[:, 3:4])[:4].detach().cpu() + 1) / 2
            grid = torch.cat([(target[:4].cpu() + 1) / 2, (pred[:4].cpu() + 1) / 2, seal_vis, mask_vis], dim=0)
            sample_path = sample_dir / f"epoch_{epoch:04d}.png"
            save_image(grid, sample_path, nrow=4)
    generator.train()
    return {k: v / max(n, 1) for k, v in totals.items()}, sample_path


def _save_checkpoint(path: Path, model_name: str, epoch: int, cfg: DictConfig, **state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "epoch": epoch, "cfg": OmegaConf.to_container(cfg, resolve=True), **state}, path)


def _generator_checkpoint_state(model: nn.Module, model_name: str) -> tuple[dict[str, torch.Tensor], bool]:
    if model_name not in {"dino_decoder", "sam_decoder", "dpt_decoder"}:
        return model.state_dict(), False
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    slim_state = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("encoder.") or name in trainable_names
    }
    return slim_state, True


def _make_adversarial_models(cfg: DictConfig):
    if cfg.model.name == "pix2pix":
        return make_pix2pix(cfg)
    if cfg.model.name == "resnet_unet":
        from .models.pix2pix import PatchDiscriminator

        generator = make_resnet_unet(cfg)
        discriminator = PatchDiscriminator(cfg.model.in_channels + cfg.model.out_channels, cfg.model.base_channels)
        return generator, discriminator
    raise ValueError(f"Unknown adversarial model: {cfg.model.name}")


def train_adversarial_reconstructor(cfg: DictConfig, device: torch.device) -> None:
    wandb_run = _start_wandb(cfg)
    train_ds, val_ds = build_train_val_datasets(cfg)
    train_loader = _loader(train_ds, cfg, shuffle=True)
    val_loader = _loader(val_ds, cfg, shuffle=False)
    generator, discriminator = _make_adversarial_models(cfg)
    generator.to(device)
    discriminator.to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=float(cfg.train.lr), betas=(float(cfg.train.beta1), float(cfg.train.beta2)))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=float(cfg.train.lr), betas=(float(cfg.train.beta1), float(cfg.train.beta2)))
    bce = nn.BCEWithLogitsLoss()

    selection_mode = str(cfg.train.selection_mode)
    selection_metric = str(cfg.train.selection_metric)
    best_metric = float("inf") if selection_mode == "min" else -float("inf")
    history: list[dict] = []
    for epoch in range(1, int(cfg.train.max_epochs) + 1):
        generator.train()
        discriminator.train()
        train_totals = {
            "loss_g": 0.0,
            "loss_d": 0.0,
            "gan_loss": 0.0,
            "l1_loss": 0.0,
            "bce_loss": 0.0,
            "dice_loss": 0.0,
            "skeleton_cldice_loss": 0.0,
            "skeleton_l1_loss": 0.0,
        }
        train_steps = 0
        pbar = tqdm(train_loader, desc=f"{cfg.model.name} epoch {epoch}")
        for batch in pbar:
            seal = batch["input"].to(device)
            schema = batch["schema"].to(device)
            weights = batch["loss_weight"].to(device)

            fake = generator(seal)
            pred_real = discriminator(seal, schema)
            pred_fake = discriminator(seal, fake.detach())
            real_loss = bce(pred_real, torch.ones_like(pred_real))
            fake_loss = bce(pred_fake, torch.zeros_like(pred_fake))
            loss_d = 0.5 * (real_loss + fake_loss)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            pred_fake_for_g = discriminator(seal, fake)
            gan_loss = bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            l1_loss = weighted_l1(fake, schema, weights, float(cfg.model.foreground_weight))
            bce_loss, dice_loss = mask_losses(fake, schema, weights)
            skeleton_cldice_loss, skeleton_l1_loss = skeleton_losses(
                fake,
                schema,
                weights,
                int(cfg.model.get("skeleton_iterations", 20)),
            )
            loss_g = (
                float(cfg.model.gan_loss_weight) * gan_loss
                + float(cfg.model.lambda_l1) * l1_loss
                + float(cfg.model.bce_loss_weight) * bce_loss
                + float(cfg.model.dice_loss_weight) * dice_loss
                + float(cfg.model.get("skeleton_cldice_loss_weight", 0.0)) * skeleton_cldice_loss
                + float(cfg.model.get("skeleton_l1_loss_weight", 0.0)) * skeleton_l1_loss
            )
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            train_steps += 1
            train_totals["loss_g"] += float(loss_g.item())
            train_totals["loss_d"] += float(loss_d.item())
            train_totals["gan_loss"] += float(gan_loss.item())
            train_totals["l1_loss"] += float(l1_loss.item())
            train_totals["bce_loss"] += float(bce_loss.item())
            train_totals["dice_loss"] += float(dice_loss.item())
            train_totals["skeleton_cldice_loss"] += float(skeleton_cldice_loss.item())
            train_totals["skeleton_l1_loss"] += float(skeleton_l1_loss.item())
            pbar.set_postfix(loss_g=f"{loss_g.item():.3f}", loss_d=f"{loss_d.item():.3f}")

        train_metrics = {f"train/{key}": value / max(train_steps, 1) for key, value in train_totals.items()}
        val_metrics, sample_path = _validate_generator(generator, val_loader, device, cfg, epoch)
        row = {"epoch": epoch, **train_metrics, **{f"val/{key}": value for key, value in val_metrics.items()}}
        history.append(row)
        print(row)
        pd.DataFrame(history).to_csv(Path(cfg.output_dir) / "history.csv", index=False)
        _save_checkpoint(
            Path(cfg.output_dir) / "checkpoints" / "last.pth",
            cfg.model.name,
            epoch,
            cfg,
            generator=generator.state_dict(),
            discriminator=discriminator.state_dict(),
            opt_g=opt_g.state_dict(),
            opt_d=opt_d.state_dict(),
        )
        if selection_metric not in val_metrics:
            raise KeyError(f"Selection metric '{selection_metric}' not found in validation metrics: {sorted(val_metrics)}")
        if _is_better_metric(float(val_metrics[selection_metric]), best_metric, selection_mode):
            best_metric = float(val_metrics[selection_metric])
            _save_checkpoint(Path(cfg.output_dir) / "checkpoints" / "best.pth", cfg.model.name, epoch, cfg, generator=generator.state_dict())
        if epoch % int(cfg.train.save_every) == 0:
            _save_checkpoint(Path(cfg.output_dir) / "checkpoints" / f"epoch_{epoch}.pth", cfg.model.name, epoch, cfg, generator=generator.state_dict())
        log_payload = {**train_metrics, **{f"val/{key}": value for key, value in val_metrics.items()}, "epoch": epoch, "best_metric": best_metric}
        if sample_path is not None and bool(cfg.train.wandb.log_samples):
            log_payload["samples/val_grid"] = _wandb_image(sample_path, f"{cfg.model.name} epoch {epoch}")
        _wandb_log(wandb_run, log_payload, step=epoch)
    if wandb_run is not None:
        wandb_run.finish()


def train_supervised_reconstructor(cfg: DictConfig, device: torch.device) -> None:
    wandb_run = _start_wandb(cfg)
    train_ds, val_ds = build_train_val_datasets(cfg)
    train_loader = _loader(train_ds, cfg, shuffle=True)
    val_loader = _loader(val_ds, cfg, shuffle=False)
    if cfg.model.name == "dino_decoder":
        model = make_dino_decoder(cfg)
    elif cfg.model.name == "sam_decoder":
        model = make_sam_decoder(cfg)
    elif cfg.model.name == "dpt_decoder":
        model = make_dpt_decoder(cfg)
    else:
        raise ValueError(f"Unknown supervised model: {cfg.model.name}")
    model.to(device)
    if cfg.model.name in {"dino_decoder", "sam_decoder", "dpt_decoder"}:
        encoder_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        param_groups = []
        if decoder_params:
            param_groups.append({"params": decoder_params, "lr": float(cfg.train.lr)})
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": float(cfg.model.get("encoder_lr", cfg.train.lr))})
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(cfg.train.lr))

    selection_mode = str(cfg.train.selection_mode)
    selection_metric = str(cfg.train.selection_metric)
    best_metric = float("inf") if selection_mode == "min" else -float("inf")
    history: list[dict] = []
    for epoch in range(1, int(cfg.train.max_epochs) + 1):
        model.train()
        train_totals = {
            "loss": 0.0,
            "l1_loss": 0.0,
            "bce_loss": 0.0,
            "dice_loss": 0.0,
            "skeleton_cldice_loss": 0.0,
            "skeleton_l1_loss": 0.0,
        }
        train_steps = 0
        pbar = tqdm(train_loader, desc=f"{cfg.model.name} epoch {epoch}")
        for batch in pbar:
            seal = batch["input"].to(device)
            schema = batch["schema"].to(device)
            weights = batch["loss_weight"].to(device)
            pred = model(seal)
            loss, parts = supervised_reconstruction_loss(pred, schema, weights, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_steps += 1
            for key, value in parts.items():
                train_totals[key] += float(value.item())
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_metrics = {f"train/{key}": value / max(train_steps, 1) for key, value in train_totals.items()}
        val_metrics, sample_path = _validate_generator(model, val_loader, device, cfg, epoch)
        row = {"epoch": epoch, **train_metrics, **{f"val/{key}": value for key, value in val_metrics.items()}}
        history.append(row)
        print(row)
        pd.DataFrame(history).to_csv(Path(cfg.output_dir) / "history.csv", index=False)
        generator_state, partial_generator = _generator_checkpoint_state(model, cfg.model.name)
        _save_checkpoint(
            Path(cfg.output_dir) / "checkpoints" / "last.pth",
            cfg.model.name,
            epoch,
            cfg,
            generator=generator_state,
            partial_generator=partial_generator,
            opt=optimizer.state_dict(),
        )
        if selection_metric not in val_metrics:
            raise KeyError(f"Selection metric '{selection_metric}' not found in validation metrics: {sorted(val_metrics)}")
        if _is_better_metric(float(val_metrics[selection_metric]), best_metric, selection_mode):
            best_metric = float(val_metrics[selection_metric])
            _save_checkpoint(
                Path(cfg.output_dir) / "checkpoints" / "best.pth",
                cfg.model.name,
                epoch,
                cfg,
                generator=generator_state,
                partial_generator=partial_generator,
            )
        if epoch % int(cfg.train.save_every) == 0:
            _save_checkpoint(
                Path(cfg.output_dir) / "checkpoints" / f"epoch_{epoch}.pth",
                cfg.model.name,
                epoch,
                cfg,
                generator=generator_state,
                partial_generator=partial_generator,
            )
        log_payload = {**train_metrics, **{f"val/{key}": value for key, value in val_metrics.items()}, "epoch": epoch, "best_metric": best_metric}
        if sample_path is not None and bool(cfg.train.wandb.log_samples):
            log_payload["samples/val_grid"] = _wandb_image(sample_path, f"{cfg.model.name} epoch {epoch}")
        _wandb_log(wandb_run, log_payload, step=epoch)
    if wandb_run is not None:
        wandb_run.finish()


def train_diffusion(cfg: DictConfig, device: torch.device) -> None:
    wandb_run = _start_wandb(cfg)
    train_ds, val_ds = build_train_val_datasets(cfg)
    train_loader = _loader(train_ds, cfg, shuffle=True)
    val_loader = _loader(val_ds, cfg, shuffle=False)
    model, ddpm = make_diffusion(cfg, device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr))
    best_mae = float("inf")
    history: list[dict] = []
    for epoch in range(1, int(cfg.train.max_epochs) + 1):
        model.train()
        train_loss_total = 0.0
        train_steps = 0
        pbar = tqdm(train_loader, desc=f"diffusion epoch {epoch}")
        for batch in pbar:
            seal = batch["input"].to(device)
            schema = batch["schema"].to(device)
            weights = batch["loss_weight"].to(device)
            noise = torch.randn_like(schema)
            t = torch.randint(0, ddpm.timesteps, (schema.size(0),), device=device)
            noisy_schema = ddpm.q_sample(schema, t, noise)
            pred_noise = model(noisy_schema, seal, t)
            per_item = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=(1, 2, 3))
            loss = (per_item * weights).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_steps += 1
            train_loss_total += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        sample_path = None
        if epoch % int(cfg.train.sample_every) == 0 or epoch == int(cfg.train.max_epochs):
            model.eval()
            with torch.no_grad():
                batch = next(iter(val_loader))
                seal = batch["input"].to(device)
                target = batch["schema"].to(device)
                pred = ddpm.sample(model, seal[: min(4, seal.size(0))])
                metrics = reconstruction_metrics(pred, target[: pred.size(0)], cfg.evaluate.threshold)
                sample_dir = Path(cfg.output_dir) / "samples"
                sample_dir.mkdir(parents=True, exist_ok=True)
                sample_path = sample_dir / f"epoch_{epoch:04d}.png"
                save_image(torch.cat([(target[: pred.size(0)].cpu() + 1) / 2, (pred.cpu() + 1) / 2], dim=0), sample_path)
        else:
            metrics = {"mae": float("nan"), "ssim": float("nan"), "binary_iou": float("nan")}
        train_metrics = {"train/loss": train_loss_total / max(train_steps, 1)}
        row = {"epoch": epoch, **train_metrics, **{f"val/{key}": value for key, value in metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(Path(cfg.output_dir) / "history.csv", index=False)
        _save_checkpoint(Path(cfg.output_dir) / "checkpoints" / "last.pth", "diffusion", epoch, cfg, model=model.state_dict(), opt=opt.state_dict())
        if metrics["mae"] == metrics["mae"] and metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            _save_checkpoint(Path(cfg.output_dir) / "checkpoints" / "best.pth", "diffusion", epoch, cfg, model=model.state_dict())
        log_payload = {**train_metrics, **{f"val/{key}": value for key, value in metrics.items()}, "epoch": epoch, "best_mae": best_mae}
        if sample_path is not None and bool(cfg.train.wandb.log_samples):
            log_payload["samples/val_grid"] = _wandb_image(sample_path, f"diffusion epoch {epoch}")
        _wandb_log(wandb_run, log_payload, step=epoch)
    if wandb_run is not None:
        wandb_run.finish()


def train(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    if cfg.model.name not in {"stable_diffusion_controlnet", "stable_diffusion_lora"}:
        _validate_input_channels(cfg)
    create_splits(cfg)
    device = get_device(cfg.device)
    save_json(Path(cfg.output_dir) / "resolved_config.json", OmegaConf.to_container(cfg, resolve=True))
    if cfg.model.name in {"pix2pix", "resnet_unet"}:
        train_adversarial_reconstructor(cfg, device)
    elif cfg.model.name in {"dino_decoder", "sam_decoder", "dpt_decoder"}:
        train_supervised_reconstructor(cfg, device)
    elif cfg.model.name == "diffusion":
        train_diffusion(cfg, device)
    elif cfg.model.name in {"stable_diffusion_controlnet", "stable_diffusion_lora"}:
        wandb_run = _start_wandb(cfg)
        train_stable_controlnet(
            cfg,
            device,
            wandb_run=wandb_run,
            wandb_log=_wandb_log,
            wandb_image=_wandb_image,
        )
        if wandb_run is not None:
            wandb_run.finish()
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")
    _evaluate_after_training(cfg)


def _evaluate_after_training(cfg: DictConfig) -> None:
    eval_after = cfg.train.get("eval_after")
    if eval_after is None or not bool(eval_after.get("enabled", False)):
        return
    checkpoint = str(eval_after.get("checkpoint", "best"))
    checkpoint_path = Path(cfg.output_dir) / "checkpoints" / f"{checkpoint}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Post-training evaluation checkpoint not found: {checkpoint_path}")

    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    eval_cfg.evaluate.checkpoint_path = str(checkpoint_path)
    eval_cfg.train.checkpoint_path = None
    eval_output_dir = eval_after.get("output_dir")
    if eval_output_dir is None:
        eval_output_dir = str(Path(cfg.output_dir) / f"eval_{checkpoint}")
    eval_cfg.output_dir = str(eval_output_dir)
    Path(eval_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    from .evaluate import evaluate

    print(f"Running post-training evaluation: {checkpoint_path} -> {eval_cfg.output_dir}")
    evaluate(eval_cfg)


def smoke_test(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    validate_paired_metadata(cfg)
    create_splits(cfg)
    cfg.data.limit_train = 4
    cfg.data.limit_val = 2
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.train.max_epochs = 1
    cfg.train.max_eval_batches = 1
    train(cfg)
