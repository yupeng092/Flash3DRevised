#!/usr/bin/env python3
"""CPU training entry that bypasses @hydra.main (incompatible with Python 3.14's
argparse).  Composes the config via hydra.compose and calls train.main().

Usage:
  python scripts/train_cpu.py +experiment=layered_re10k_cpu_debug_v1 \
      dataset.data_path=data/RealEstate10K optimiser.num_epochs=1 ...
All Hydra overrides work as usual; they are parsed from sys.argv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Redirect torch.hub cache to a project-local writable dir so VGG16 (LPIPS) and
# any other hub assets download without hitting the read-only global ~/.cache.
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "weights" / "torch_hub_cache"))

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig


def main() -> None:
    # Collect Hydra-style overrides from argv (everything after the script name).
    overrides = sys.argv[1:]
    config_dir = str(PROJECT_ROOT / "configs")

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)

    import train as train_module
    train_module.main(cfg)


if __name__ == "__main__":
    main()
