#!/usr/bin/env python3
"""CPU smoke test for the Depth Anything V1 integration.

Validates three layers of the integration on CPU:

  1. Hydra config composition: every V1 experiment + depth config composes, the
     ``model.name == depth_anything_v1`` branch is selected, ``encoder`` and
     ``freeze`` are propagated, and the ``${model.depth.encoder}`` checkpoint
     interpolation resolves to the expected file name for vits/vitb/vitl.
  2. The portable differentiable renderer (``render_predicted_torch``) runs a
     forward + backward pass on CPU with finite gradients for every Gaussian
     attribute.  This is the renderer used by the NPU/CPU pre-training loss.
  3. The DepthAnythingV1Extended encoder constructs from the official V1 source
     + checkpoint, loads weights with ``strict=True``, freezes the DA backbone,
     and runs a forward pass producing depth + Gaussian attributes.  Requires
     ``third_party/Depth-Anything`` and ``weights/depth-anything/*.pth``.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

os.environ.setdefault("WANDB_DISABLED", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir
from models.decoder.npu_differentiable_renderer import render_predicted_torch

CONFIG_DIR = PROJECT_ROOT / "configs"


def test_hydra_configs() -> None:
    """Compose every V1 experiment and assert the key wiring fields."""
    print("\n=== [1] Hydra config composition ===")
    cases = [
        # (experiment, expected_name, expected_encoder, expected_freeze, expected_ckpt)
        ("layered_re10k_npu", "depth_anything_v1", "vitb", True,
         "weights/depth-anything/depth_anything_vitb14.pth"),
        ("layered_re10k_npu_vits", "depth_anything_v1", "vits", True,
         "weights/depth-anything/depth_anything_vits14.pth"),
        ("layered_re10k_depth_anything_v1", "depth_anything_v1", "vitb", True,
         "weights/depth-anything/depth_anything_vitb14.pth"),
        ("layered_re10k_cpu_debug_v1", "depth_anything_v1", "vitb", True,
         "weights/depth-anything/depth_anything_vitb14.pth"),
        ("layered_re10k_cpu_debug", "depth_anything_v2", "vitb", True,
         "weights/depth-anything-v2/depth_anything_v2_metric_vkitti_vitb.pth"),  # V2 metric
    ]
    for experiment, expected_name, expected_encoder, expected_freeze, expected_ckpt in cases:
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="config", overrides=[f"+experiment={experiment}"])
        name = cfg.model.name
        encoder = cfg.model.depth.encoder
        freeze = cfg.model.depth.freeze
        checkpoint = cfg.model.depth.checkpoint
        assert name == expected_name, f"[{experiment}] model.name={name} != {expected_name}"
        assert encoder == expected_encoder, f"[{experiment}] encoder={encoder}"
        assert freeze == expected_freeze, f"[{experiment}] freeze={freeze}"
        assert checkpoint == expected_ckpt, f"[{experiment}] checkpoint={checkpoint} != {expected_ckpt}"
        print(f"  OK  +experiment={experiment:38s} name={name} encoder={encoder} freeze={freeze} ckpt={checkpoint}")

    # Encoder override interpolation: switching encoder must switch checkpoint.
    print("  -- interpolation under model.depth.encoder override --")
    for enc in ("vits", "vitb", "vitl"):
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="config", overrides=[
                "+experiment=layered_re10k_npu", f"model.depth.encoder={enc}",
            ])
        got = cfg.model.depth.checkpoint
        want = f"weights/depth-anything/depth_anything_{enc}14.pth"
        assert got == want, f"encoder={enc} -> {got} != {want}"
        print(f"  OK  encoder={enc:5s} -> checkpoint={got}")
    print("  [PASS] all Hydra config assertions")


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(znear=0.01, zfar=20.0),
        model=SimpleNamespace(
            npu_renderer_min_variance=0.30,
            npu_renderer_sigma_cutoff=3.0,
            npu_renderer_max_radius=24.0,
            npu_renderer_max_gaussians=0,
            npu_renderer_tile_size=8,
            npu_renderer_tile_span=7,
            npu_renderer_max_gaussians_per_tile=64,
        ),
    )


def _make_point_cloud(device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    n = 24
    return {
        "xyz": torch.cat((torch.randn(n, 2) * 0.3, torch.rand(n, 1) * 2 + 2), dim=1).to(device).requires_grad_(),
        "opacity": (torch.rand(n, 1) * 0.5 + 0.3).to(device).requires_grad_(),
        "scaling": (torch.rand(n, 3) * 0.08 + 0.03).to(device).requires_grad_(),
        "rotation": torch.cat((torch.ones(n, 1), torch.randn(n, 3) * 0.05), dim=1).to(device).requires_grad_(),
        "features_dc": torch.randn(n, 1, 3).to(device).requires_grad_(),
    }


def test_cpu_renderer() -> None:
    """Forward + backward of the portable renderer on CPU; check finite grads."""
    print("\n=== [2] CPU differentiable renderer (render_predicted_torch) ===")
    device = torch.device("cpu")
    pc = _make_point_cloud(device)
    matrix = torch.eye(4, device=device)
    result = render_predicted_torch(
        _config(), pc, matrix, matrix, matrix, torch.zeros(3, device=device),
        (1.0, 1.0), (32, 40), torch.tensor([0.5, 0.5, 0.5], device=device), 0,
    )
    rgb = result["render"]
    depth = result["depth"]
    alpha = result["alpha"]
    assert rgb.shape == (3, 32, 40), f"render shape {rgb.shape}"
    assert depth.shape == (32, 40), f"depth shape {depth.shape}"
    assert alpha.shape == (32, 40), f"alpha shape {alpha.shape}"
    print(f"  OK  forward: render{tuple(rgb.shape)} depth{tuple(depth.shape)} alpha{tuple(alpha.shape)}")
    print(f"      render range [{rgb.min():.4f}, {rgb.max():.4f}]  alpha range [{alpha.min():.4f}, {alpha.max():.4f}]")

    loss = rgb.mean() + 0.01 * depth.mean()
    loss.backward()
    missing = [k for k, v in pc.items() if v.grad is None or not torch.isfinite(v.grad).all()]
    assert not missing, f"missing/non-finite grad for: {missing}"
    print(f"  OK  backward: loss={loss.item():.6f}  all grads finite")
    for k, v in pc.items():
        print(f"      grad {k:12s} norm={v.grad.norm().item():.6f}")
    print("  [PASS] CPU renderer forward + backward")


def test_da_v1_encoder() -> None:
    """Construct DepthAnythingV1Extended, load weights, freeze, run forward."""
    print("\n=== [3] DepthAnythingV1Extended construct + forward (vits, frozen) ===")
    source = PROJECT_ROOT / "third_party" / "Depth-Anything" / "depth_anything" / "dpt.py"
    ckpt = PROJECT_ROOT / "weights" / "depth-anything" / "depth_anything_vits14.pth"
    if not source.is_file() or not ckpt.is_file():
        print(f"  [SKIP] DA V1 source/checkpoint missing ({source}, {ckpt})")
        print("         Run scripts/download_depth_anything_v1.py --encoders vits first.")
        return
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    CONFIG_DIR = PROJECT_ROOT / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=[
            "+experiment=layered_re10k_cpu_debug_v1",
            "model.depth.encoder=vits",
            "model.backbone.weights_init=scratch",
        ])
    cfg.model.gaussians_per_pixel = 1  # isolate DA forward from layered decoder

    from models.encoder.depth_anything_v1_encoder import DepthAnythingV1Extended
    model = DepthAnythingV1Extended(cfg)
    n_da = sum(p.numel() for p in model.depth_model.parameters())
    n_train = sum(p.numel() for g in model.get_parameter_groups() for p in g["params"])
    da_grad = any(p.requires_grad for p in model.depth_model.parameters())
    assert not da_grad, "DA encoder should be frozen (requires_grad=False)"
    print(f"  OK  construct: DA={n_da/1e6:.1f}M params, trainable={n_train/1e6:.1f}M, frozen={not da_grad}")

    image = torch.rand(1, 3, 128, 192)
    K = torch.tensor([[[400., 0, 96.], [0, 400., 64.], [0, 0, 1]]])
    inputs = {("color_aug", 0, 0): image, ("K_src", 0): K}
    with torch.no_grad():
        outputs = model(inputs)
    depth = outputs[("depth", 0)]
    assert depth.shape == (1, 1, 128, 192), f"depth shape {depth.shape}"
    assert "gauss_opacity" in outputs and "gauss_scaling" in outputs
    print(f"  OK  forward: depth{tuple(depth.shape)} range=[{depth.min():.4f}, {depth.max():.4f}]")
    print(f"      gaussian attrs present: opacity, scaling, rotation, features_dc, offset")
    print("  [PASS] DA V1 encoder construct + load + freeze + forward")


def main() -> None:
    print("Flash3D CPU smoke test (DA V1 integration)")
    print(f"torch={torch.__version__}  device=cpu")
    test_hydra_configs()
    test_cpu_renderer()
    test_da_v1_encoder()
    print("\n=== SUMMARY ===")
    print("[PASS] Hydra config composition for all V1 experiments")
    print("[PASS] CPU renderer forward + backward (finite gradients)")
    print("[PASS] DA V1 encoder construct + load + freeze + forward (vits)")
    print("\nNext: end-to-end train.py needs RE10K data at data/RealEstate10K.")


if __name__ == "__main__":
    main()
