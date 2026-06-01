from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from ..data import build_train_val_datasets
from ..metrics import reconstruction_metrics
from ..utils import save_json


def _sd_imports():
    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, StableDiffusionControlNetPipeline, UNet2DConditionModel
    from transformers import AutoTokenizer, CLIPTextModel

    return AutoencoderKL, ControlNetModel, DDPMScheduler, StableDiffusionControlNetPipeline, UNet2DConditionModel, AutoTokenizer, CLIPTextModel


def sd_condition_from_batch(
    batch: dict[str, Any],
    device: torch.device,
    condition_mode: str = "seal_mask",
    invert: bool = False,
) -> torch.Tensor:
    seal_rgb = batch["seal_rgb"].to(device)
    mask = batch["mask"].to(device)
    seal01 = ((seal_rgb[:, :1].clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
    mask01 = ((mask.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
    if condition_mode == "seal_mask":
        condition = torch.cat([seal01, mask01, seal01 * mask01], dim=1)
    elif condition_mode == "mask_only":
        condition = mask01.repeat(1, 3, 1, 1)
    elif condition_mode == "seal_only":
        condition = seal01.repeat(1, 3, 1, 1)
    else:
        raise ValueError(f"Unknown model.condition_mode: {condition_mode}")
    if invert:
        condition = 1.0 - condition
    return condition


def _schema_pixel_values(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    schema = batch["schema"].to(device)
    return schema.repeat(1, 3, 1, 1).clamp(-1, 1)


def _prompt_ids(tokenizer, prompt: str, batch_size: int, device: torch.device) -> torch.Tensor:
    tokens = tokenizer(
        [prompt] * batch_size,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokens.input_ids.to(device)


def _pil_to_schema_tensor(images: list[Image.Image], device: torch.device) -> torch.Tensor:
    tensors = []
    for image in images:
        gray = image.convert("L").resize((image.width, image.height), Image.Resampling.BICUBIC)
        arr = torch.from_numpy(__import__("numpy").array(gray)).float() / 255.0
        tensors.append(arr.unsqueeze(0))
    return (torch.stack(tensors, dim=0).to(device) * 2.0 - 1.0).clamp(-1, 1)


def _condition_to_pils(condition: torch.Tensor) -> list[Image.Image]:
    images = []
    condition = condition.detach().cpu().clamp(0, 1)
    for item in condition:
        arr = (item.permute(1, 2, 0).numpy() * 255).astype("uint8")
        images.append(Image.fromarray(arr, mode="RGB"))
    return images


def _seal_to_ip_adapter_pils(batch: dict[str, Any]) -> list[Image.Image]:
    images = []
    seal_rgb = ((batch["seal_rgb"].detach().cpu().clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
    for item in seal_rgb:
        arr = (item.permute(1, 2, 0).numpy() * 255).astype("uint8")
        images.append(Image.fromarray(arr, mode="RGB"))
    return images


def _ip_adapter_images_arg(batch: dict[str, Any], cfg: DictConfig, limit: int | None = None) -> list[list[Image.Image]] | None:
    """Diffusers expects one image list per loaded IP-Adapter."""
    if not _ip_adapter_enabled(cfg):
        return None
    images = _seal_to_ip_adapter_pils(batch)
    if limit is not None:
        images = images[:limit]
    return [images]


def _ip_adapter_enabled(cfg: DictConfig) -> bool:
    return bool(cfg.model.get("ip_adapter_enabled", False))


def _load_ip_adapter_if_needed(pipe, cfg: DictConfig) -> None:
    if not _ip_adapter_enabled(cfg):
        return
    pipe.load_ip_adapter(
        str(cfg.model.ip_adapter_model),
        subfolder=str(cfg.model.ip_adapter_subfolder),
        weight_name=str(cfg.model.ip_adapter_weight_name),
    )
    pipe.set_ip_adapter_scale(float(cfg.model.get("ip_adapter_scale", 0.7)))


def _add_lora_if_needed(unet, cfg: DictConfig) -> None:
    if not bool(cfg.model.get("train_lora", False)):
        return
    from peft import LoraConfig

    rank = int(cfg.model.get("lora_rank", 16))
    alpha = int(cfg.model.get("lora_alpha", rank))
    unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
        adapter_name="schema_lora",
    )


def _load_lora_if_present(pipe, checkpoint: dict[str, Any] | None) -> None:
    if not checkpoint:
        return
    lora_dir = checkpoint.get("lora_dir")
    if not lora_dir:
        return
    lora_dir = Path(lora_dir)
    if not lora_dir.exists():
        sibling = Path(checkpoint.get("_checkpoint_path", "")).parent / "lora_best"
        if sibling.exists():
            lora_dir = sibling
    if lora_dir.exists():
        weight_name = "pytorch_lora_weights.safetensors" if (lora_dir / "pytorch_lora_weights.safetensors").exists() else None
        if weight_name:
            pipe.unet.load_lora_adapter(lora_dir, prefix=None, weight_name=weight_name, adapter_name="schema_lora")
        else:
            pipe.unet.load_lora_adapter(lora_dir, prefix=None, adapter_name="schema_lora")
        pipe.unet.set_adapter("schema_lora")


def _save_lora_if_needed(unet, cfg: DictConfig, directory: Path) -> str | None:
    if not bool(cfg.model.get("train_lora", False)):
        return None
    directory.mkdir(parents=True, exist_ok=True)
    unet.save_lora_adapter(directory, adapter_name="schema_lora", safe_serialization=True)
    return str(directory.resolve())


class StableControlNetSampler:
    def __init__(self, checkpoint_cfg: DictConfig, controlnet_dir: str | Path, device: torch.device, checkpoint: dict[str, Any] | None = None):
        _, ControlNetModel, _, StableDiffusionControlNetPipeline, _, _, _ = _sd_imports()
        controlnet = ControlNetModel.from_pretrained(controlnet_dir)
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            checkpoint_cfg.model.base_model,
            controlnet=controlnet,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        _load_ip_adapter_if_needed(self.pipeline, checkpoint_cfg)
        _load_lora_if_present(self.pipeline, checkpoint)
        self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline_config = checkpoint_cfg
        self.prompt = str(checkpoint_cfg.model.prompt)
        self.negative_prompt = str(checkpoint_cfg.model.negative_prompt)
        self.condition_mode = str(checkpoint_cfg.model.get("condition_mode", "seal_mask"))
        self.condition_invert = bool(checkpoint_cfg.model.get("condition_invert", False))
        self.num_inference_steps = int(checkpoint_cfg.model.num_inference_steps)
        self.guidance_scale = float(checkpoint_cfg.model.guidance_scale)
        self.controlnet_conditioning_scale = float(checkpoint_cfg.model.get("controlnet_conditioning_scale", 1.0))
        self.device = device

    @torch.no_grad()
    def sample(self, batch: dict[str, Any], device: torch.device | None = None) -> torch.Tensor:
        device = device or self.device
        condition = sd_condition_from_batch(batch, device, self.condition_mode, self.condition_invert)
        images = self.pipeline(
            prompt=[self.prompt] * condition.size(0),
            negative_prompt=[self.negative_prompt] * condition.size(0),
            image=_condition_to_pils(condition),
            ip_adapter_image=_ip_adapter_images_arg(batch, self.pipeline_config),
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
        ).images
        return _pil_to_schema_tensor(images, device)


def make_stable_controlnet_sampler(cfg: DictConfig, checkpoint_path: str | Path, device: torch.device) -> StableControlNetSampler:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint["_checkpoint_path"] = str(checkpoint_path)
        checkpoint_cfg = OmegaConf.create(checkpoint["cfg"])
        controlnet_dir = Path(checkpoint["controlnet_dir"])
        if not controlnet_dir.exists() or not (controlnet_dir / "config.json").exists():
            sibling_best = checkpoint_path.parent / "controlnet_best"
            sibling_last = checkpoint_path.parent / f"controlnet_step_{checkpoint.get('step')}"
            if (sibling_best / "config.json").exists():
                controlnet_dir = sibling_best
            elif (sibling_last / "config.json").exists():
                controlnet_dir = sibling_last
    else:
        checkpoint = None
        checkpoint_cfg = cfg
        controlnet_dir = checkpoint_path
    return StableControlNetSampler(checkpoint_cfg, controlnet_dir, device, checkpoint)


@torch.no_grad()
def _validate_controlnet(
    cfg: DictConfig,
    controlnet,
    tokenizer,
    text_encoder,
    vae,
    unet,
    noise_scheduler,
    val_loader: DataLoader,
    device: torch.device,
    step: int,
) -> tuple[dict[str, float], Path | None]:
    _, _, _, StableDiffusionControlNetPipeline, _, _, _ = _sd_imports()
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.model.base_model,
        controlnet=controlnet,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    _load_ip_adapter_if_needed(pipe, cfg)
    pipe.set_progress_bar_config(disable=True)
    batch = next(iter(val_loader))
    condition = sd_condition_from_batch(
        batch,
        device,
        str(cfg.model.get("condition_mode", "seal_mask")),
        bool(cfg.model.get("condition_invert", False)),
    )
    target = batch["schema"].to(device)
    images = pipe(
        prompt=[str(cfg.model.prompt)] * min(4, condition.size(0)),
        negative_prompt=[str(cfg.model.negative_prompt)] * min(4, condition.size(0)),
        image=_condition_to_pils(condition[:4]),
        ip_adapter_image=_ip_adapter_images_arg(batch, cfg, limit=4),
        num_inference_steps=int(cfg.model.num_inference_steps),
        guidance_scale=float(cfg.model.guidance_scale),
        controlnet_conditioning_scale=float(cfg.model.get("controlnet_conditioning_scale", 1.0)),
    ).images
    pred = _pil_to_schema_tensor(images, device)
    metrics = reconstruction_metrics(pred, target[: pred.size(0)], cfg.evaluate.threshold)
    sample_dir = Path(cfg.output_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"step_{step:06d}.png"
    seal_vis = (batch["seal_rgb"][: pred.size(0), :1].cpu() + 1) / 2
    mask_vis = (batch["mask"][: pred.size(0)].cpu() + 1) / 2
    grid = torch.cat([(target[: pred.size(0)].cpu() + 1) / 2, (pred.cpu() + 1) / 2, seal_vis, mask_vis], dim=0)
    save_image(grid, sample_path, nrow=pred.size(0))
    del pipe
    torch.cuda.empty_cache()
    return metrics, sample_path


def train_stable_controlnet(cfg: DictConfig, device: torch.device, wandb_run=None, wandb_log=None, wandb_image=None) -> None:
    AutoencoderKL, ControlNetModel, DDPMScheduler, StableDiffusionControlNetPipeline, UNet2DConditionModel, AutoTokenizer, CLIPTextModel = _sd_imports()
    train_ds, val_ds = build_train_val_datasets(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.model.train_batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=min(4, int(cfg.model.train_batch_size) * 4), shuffle=False, num_workers=0)

    base_model = str(cfg.model.base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet").to(device)
    controlnet_model = str(cfg.model.get("controlnet_model", "from_unet"))
    if controlnet_model and controlnet_model != "from_unet":
        controlnet = ControlNetModel.from_pretrained(controlnet_model).to(device)
    else:
        controlnet = ControlNetModel.from_unet(unet).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        controlnet=controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    _load_ip_adapter_if_needed(pipe, cfg)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    controlnet = pipe.controlnet
    noise_scheduler = pipe.scheduler

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    _add_lora_if_needed(unet, cfg)
    controlnet.train()
    trainable_params = list(controlnet.parameters()) + [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg.model.learning_rate))

    save_json(Path(cfg.output_dir) / "resolved_config.json", OmegaConf.to_container(cfg, resolve=True))
    max_steps = int(cfg.model.max_train_steps)
    grad_accum = int(cfg.model.gradient_accumulation_steps)
    validation_every = int(cfg.model.validation_every)
    save_every = int(cfg.model.save_every_steps)
    global_step = 0
    history: list[dict[str, float]] = []
    best_iou = -1.0

    pbar = tqdm(total=max_steps, desc="stable_diffusion_controlnet")
    while global_step < max_steps:
        for batch in train_loader:
            with torch.no_grad():
                pixel_values = _schema_pixel_values(batch, device)
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                input_ids = _prompt_ids(tokenizer, str(cfg.model.prompt), latents.size(0), device)
                encoder_hidden_states = text_encoder(input_ids)[0]
                condition = sd_condition_from_batch(
                    batch,
                    device,
                    str(cfg.model.get("condition_mode", "seal_mask")),
                    bool(cfg.model.get("condition_invert", False)),
                )
                added_cond_kwargs = None
                if _ip_adapter_enabled(cfg):
                    image_embeds = pipe.prepare_ip_adapter_image_embeds(
                        _ip_adapter_images_arg(batch, cfg),
                        None,
                        device,
                        1,
                        False,
                    )
                    added_cond_kwargs = {"image_embeds": image_embeds}

            down_res, mid_res = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=condition,
                return_dict=False,
            )
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            loss = F.mse_loss(model_pred.float(), noise.float()) / grad_accum
            loss.backward()

            if (global_step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")
            row = {"step": global_step, "train/loss": float(loss.item() * grad_accum)}

            sample_path = None
            if global_step % validation_every == 0 or global_step == max_steps:
                controlnet.eval()
                metrics, sample_path = _validate_controlnet(
                    cfg, controlnet, tokenizer, text_encoder, vae, unet, noise_scheduler, val_loader, device, global_step
                )
                controlnet.train()
                row.update({f"val/{key}": value for key, value in metrics.items()})
                if metrics["binary_iou"] > best_iou:
                    best_iou = metrics["binary_iou"]
                    best_dir = Path(cfg.output_dir) / "checkpoints" / "controlnet_best"
                    controlnet.save_pretrained(best_dir)
                    lora_dir = _save_lora_if_needed(unet, cfg, Path(cfg.output_dir) / "checkpoints" / "lora_best")
                    torch.save(
                        {
                            "model_name": "stable_diffusion_controlnet",
                            "cfg": OmegaConf.to_container(cfg, resolve=True),
                            "controlnet_dir": str(best_dir.resolve()),
                            "lora_dir": lora_dir,
                            "step": global_step,
                        },
                        Path(cfg.output_dir) / "checkpoints" / "best.pth",
                    )

            if global_step % save_every == 0 or global_step == max_steps:
                ckpt_dir = Path(cfg.output_dir) / "checkpoints" / f"controlnet_step_{global_step}"
                controlnet.save_pretrained(ckpt_dir)
                lora_dir = _save_lora_if_needed(unet, cfg, Path(cfg.output_dir) / "checkpoints" / f"lora_step_{global_step}")
                torch.save(
                    {
                        "model_name": "stable_diffusion_controlnet",
                        "cfg": OmegaConf.to_container(cfg, resolve=True),
                        "controlnet_dir": str(ckpt_dir.resolve()),
                        "lora_dir": lora_dir,
                        "step": global_step,
                    },
                    Path(cfg.output_dir) / "checkpoints" / "last.pth",
                )

            history.append(row)
            pd.DataFrame(history).to_csv(Path(cfg.output_dir) / "history.csv", index=False)
            if wandb_log is not None:
                payload = {**row, "best_iou": best_iou}
                if sample_path is not None and wandb_image is not None and bool(cfg.train.wandb.log_samples):
                    payload["samples/val_grid"] = wandb_image(sample_path, f"stable diffusion step {global_step}")
                wandb_log(wandb_run, payload, step=global_step)
            if global_step >= max_steps:
                break
    pbar.close()
