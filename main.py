from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.evaluate import evaluate
from src.generate import generate
from src.pseudo_label import pseudo_label
from src.splits import create_splits
from src.train import smoke_test, train


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if cfg.mode == "create_splits":
        create_splits(cfg)
    elif cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "pseudo_label":
        pseudo_label(cfg)
    elif cfg.mode == "generate":
        generate(cfg)
    elif cfg.mode == "evaluate":
        evaluate(cfg)
    elif cfg.mode == "smoke_test":
        smoke_test(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
