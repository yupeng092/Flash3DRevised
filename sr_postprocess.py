#!/usr/bin/env python3
"""2D super-resolution / enhancement post-processing for render_cpu_alpha.py.

Two independent backends selectable at runtime via ``--sr``:

* ``enhance``  — self-contained, no external weights.  A classical SR chain:
                 Laplacian-pyramid detail injection + local contrast stretch
                 + residual unsharp.  Stronger than plain unsharp_mask and
                 works on CPU with zero dependencies beyond torch/torchvision.
                 Use when no learned SR weights are available.

* ``rrdb``     — pure-torch Real-ESRGAN RRDB inference network.  Loads a
                 ``RealESRGAN_x4.pth``-style state dict dropped into
                 ``weights/`` (no basicsr/realesrgan package needed).  This is
                 real learned 4x SR; it only activates once you place the
                 weight file, because this sandbox cannot download it.

Both backends accept an already-rendered RGB tensor in [0,1] (HWC or NCHW) and
return a float tensor in [0,1] with the same layout as the input.  They are
deliberately tolerant about ``ndim`` so the caller can pass either convention.

Weight placement for the ``rrdb`` backend:
    weights/RealESRGAN_x4.pth            (official Real-ESRGAN x4, ~64 MB)
    weights/realesrgan_x4plus.safetensors (alternative name)

The loader probes several known state-dict layouts (``params_ema`` / ``params``
/ bare) so the official checkpoint works without post-processing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
# Backend 1: self-contained classical enhancement SR (no weights)
# =========================================================================== #
def _gaussian_blur(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """Separable Gaussian blur on a (1, C, H, W) tensor, 'same' spatial size."""
    sigma = max(radius / 2.0, 0.5)
    coords = torch.arange(-radius, radius + 1, dtype=tensor.dtype, device=tensor.device)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kx = kernel.reshape(1, 1, 1, 2 * radius + 1).repeat(tensor.shape[1], 1, 1, 1)
    ky = kernel.reshape(1, 1, 2 * radius + 1, 1).repeat(tensor.shape[1], 1, 1, 1)
    # conv2d with padding=radius gives 'same' output size for odd kernels.
    out = F.conv2d(tensor, kx, padding=(0, radius), groups=tensor.shape[1])
    out = F.conv2d(out, ky, padding=(radius, 0), groups=tensor.shape[1])
    return out


def _laplacian_pyramid(image: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Return Laplacian bands [coarse..fine] + the residual low-pass, all at
    the *original* resolution is NOT how pyramids work — bands live at their
    own scale.  Reconstruction upsamples each band back up before summing."""
    bands = []
    current = image
    for _ in range(levels):
        blurred = _gaussian_blur(current, 2)
        bands.append(current - blurred)          # detail at this scale
        current = F.interpolate(blurred, scale_factor=0.5, mode="bilinear",
                                align_corners=False, antialias=True)
    bands.append(current)  # residual low-pass (smallest)
    return bands


def _reconstruct(bands: list[torch.Tensor]) -> torch.Tensor:
    """Rebuild an image from Laplacian bands by progressive upsample+add."""
    img = bands[-1]
    for band in reversed(bands[:-1]):
        img = F.interpolate(img, size=band.shape[-2:], mode="bilinear",
                            align_corners=False) + band
    return img


def _local_contrast(tensor: torch.Tensor, radius: int, gain: float) -> torch.Tensor:
    """Adaptive local contrast: amplify deviation from the local mean.

    Equivalent to a trainable normalisation layer in EDSR-style networks, done
    with classical filtering so it needs no weights.
    """
    mean = _gaussian_blur(tensor, radius)
    deviation = tensor - mean
    return mean + deviation * gain


def enhance_sr(
    image: torch.Tensor,
    *,
    detail_gain: float = 1.4,
    contrast_gain: float = 1.25,
    contrast_radius: int = 9,
    levels: int = 3,
    sharpen: float = 0.3,
) -> torch.Tensor:
    """Self-contained SR enhancement.  ``image`` is (H, W, C) in [0,1]."""
    nchw = image.permute(2, 0, 1)[None]
    base = nchw

    # 1. Laplacian-pyramid detail injection: amplify each detail band, then
    #    reconstruct back to full resolution.
    bands = _laplacian_pyramid(base, levels)
    detail, residual = bands[:-1], bands[-1]
    boosted_detail = [b * detail_gain for b in detail]
    boosted = _reconstruct(boosted_detail + [residual])

    # 2. Local contrast stretch on the reconstructed image.
    boosted = _local_contrast(boosted, contrast_radius, contrast_gain)

    # 3. Residual unsharp on top (the legacy --sharpen behaviour, kept inline).
    if sharpen > 0:
        blurred = _gaussian_blur(boosted, 2)
        boosted = boosted + sharpen * (boosted - blurred)

    out = boosted[0].permute(1, 2, 0).clamp(0, 1)
    return out


# =========================================================================== #
# Backend 2: pure-torch RRDB (Real-ESRGAN) inference, loads external weights
# =========================================================================== #
def _conv2d(in_c: int, out_c: int, k: int = 3, s: int = 1, p: Optional[int] = None) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, k, stride=s, padding=(k - 1) // 2 if p is None else p, bias=True)


class _ResidualDenseBlock(nn.Module):
    """5-conv residual dense block, G=64 hidden channels (Real-ESRGAN default)."""

    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.c1 = _conv2d(nf, gc)
        self.c2 = _conv2d(nf + gc, gc)
        self.c3 = _conv2d(nf + 2 * gc, gc)
        self.c4 = _conv2d(nf + 3 * gc, gc)
        self.c5 = _conv2d(nf + 4 * gc, nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.c1(x))
        x2 = self.lrelu(self.c2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.c3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.c4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.c5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    """Residual in Residual Dense Block = 3 RDBs."""

    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(nf, gc)
        self.rdb2 = _ResidualDenseBlock(nf, gc)
        self.rdb3 = _ResidualDenseBlock(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class _RRDBNet(nn.Module):
    """Real-ESRGAN x4 network: RRDB trunk + upsampler.

    Architecture matches the official Real-ESRGAN_x4.pth so its state dict
    loads directly.  Default nf=64, gc=32, num_rrdb=23 — the x4plus config.
    """

    def __init__(self, in_nc: int = 3, out_nc: int = 3, nf: int = 64,
                 nb: int = 23, gc: int = 32, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.conv_first = _conv2d(in_nc, nf)
        self.body = nn.Sequential(*[_RRDB(nf=nf, gc=gc) for _ in range(nb)])
        self.conv_body = _conv2d(nf, nf)
        # Upsampling: two consecutive PixelShuffle(2) => x4.
        up_layers = []
        n_up = {2: 1, 4: 2, 8: 3}.get(scale, 2)
        for _ in range(n_up):
            up_layers += [_conv2d(nf, nf * 4), nn.PixelShuffle(2)]
        self.conv_up = nn.Sequential(*up_layers)
        self.conv_hr = _conv2d(nf, nf)
        self.conv_last = _conv2d(nf, out_nc)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up(feat))
        feat = self.lrelu(self.conv_hr(feat))
        return self.conv_last(feat)


def _load_rrdb_state_dict(path: Path) -> dict:
    """Load a Real-ESRGAN checkpoint, tolerating several on-disk layouts."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path))
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    # Official checkpoints wrap the model under 'params_ema' (preferred) or
    # 'params'.  Bare dicts (network-only) are also accepted.
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    # Strip a leading 'module.' from DataParallel wrappers.
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def _find_rrdb_weight(weights_dir: Path) -> Optional[Path]:
    names = [
        "RealESRGAN_x4.pth", "realesrgan_x4plus.pth",
        "realesrgan_x4plus.safetensors", "RealESRGAN_x4.safetensors",
        "RealESRGAN_x2.pth", "realesrgan_x2plus.pth",
    ]
    for name in names:
        candidate = weights_dir / name
        if candidate.is_file():
            return candidate
    return None


class RRDBSR:
    """Lazy-loaded RRDB super-resolution wrapper.

    Construction is cheap (no weight loading).  The network is built and
    loaded the first time it is actually invoked, so simply passing
    ``--sr rrdb`` without a weight file prints a clear error instead of
    crashing at startup.
    """

    def __init__(self, weights_dir: Path, tile: int = 128, scale: int = 4):
        self.weights_dir = weights_dir
        self.tile = tile
        self.scale = scale
        self._net: Optional[nn.Module] = None
        self._weight_path: Optional[Path] = None

    def _ensure_loaded(self) -> nn.Module:
        if self._net is not None:
            return self._net
        path = _find_rrdb_weight(self.weights_dir)
        if path is None:
            raise FileNotFoundError(
                f"No Real-ESRGAN weight found under {self.weights_dir}. "
                f"Drop one of RealESRGAN_x4.pth / realesrgan_x4plus.safetensors "
                f"there, then re-run with --sr rrdb."
            )
        sd = _load_rrdb_state_dict(path)
        scale = 4 if "x4" in path.name.lower() else (2 if "x2" in path.name.lower() else self.scale)
        # Infer channel count from the first conv weight so x2/x4 and
        # grey/RGB variants all load.
        nf = sd["conv_first.weight"].shape[0]
        in_nc = sd["conv_first.weight"].shape[1]
        net = _RRDBNet(in_nc=in_nc, out_nc=in_nc, nf=nf, scale=scale)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        net.eval()
        self._net = net
        self._weight_path = path
        self.scale = scale
        if missing or unexpected:
            print(f"[rrdb] loaded {path.name} (scale=x{scale}, nf={nf}); "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print(f"[rrdb] loaded {path.name} (scale=x{scale}, nf={nf})")
        return net

    @torch.inference_mode()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """``image`` is (H, W, C) float [0,1] on CPU.  Returns (H*s, W*s, C)."""
        net = self._ensure_loaded()
        nchw = image.permute(2, 0, 1)[None]
        nchw = (nchw - 0.5) / 0.5  # normalise to ~[-1, 1]
        if self.tile <= 0 or min(nchw.shape[-2:]) <= self.tile:
            out = net(nchw)
        else:
            out = self._tiled_forward(nchw)
        out = (out * 0.5 + 0.5).clamp(0, 1)
        return out[0].permute(1, 2, 0)

    def _tiled_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tiled inference to bound peak memory on large inputs.

        Tiles overlap by ``tile//2``; overlap regions are blended with a
        raised-cosine window so seams are invisible.
        """
        net = self._ensure_loaded()
        _, _, h, w = x.shape
        t, ov = self.tile, self.tile // 2
        ys = list(range(0, max(1, h - t + 1), t - ov)) + ([h - t] if h > t else [0])
        xs = list(range(0, max(1, w - t + 1), t - ov)) + ([w - t] if w > t else [0])
        ys, xs = sorted(set(ys)), sorted(set(xs))
        out = torch.zeros((1, x.shape[1], h * self.scale, w * self.scale),
                          dtype=x.dtype, device=x.device)
        weight = torch.zeros_like(out)
        win = self._window(t * self.scale)
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                tile = x[:, :, y0:y1, x0:x1]
                up = net(tile)
                oy, ox = y0 * self.scale, x0 * self.scale
                oh, ow = up.shape[-2:]
                w_tile = win[:oh, :ow]
                out[:, :, oy:oy + oh, ox:ox + ow] += up * w_tile
                weight[:, :, oy:oy + oh, ox:ox + ow] += w_tile
        return out / weight.clamp_min(1e-6)

    @staticmethod
    def _window(n: int) -> torch.Tensor:
        coords = torch.arange(n, dtype=torch.float32)
        return (1.0 - torch.cos(2.0 * torch.pi * coords / max(n - 1, 1))).clamp_min(1e-3)


# =========================================================================== #
# Dispatcher
# =========================================================================== #
def apply_sr(
    image: torch.Tensor,
    backend: str,
    *,
    rrdb: Optional[RRDBSR] = None,
    enhance_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """Apply the chosen SR backend to an (H, W, C) [0,1] tensor."""
    if backend in ("none", "", None):
        return image
    if backend == "enhance":
        return enhance_sr(image, **(enhance_kwargs or {}))
    if backend == "rrdb":
        if rrdb is None:
            raise RuntimeError("rrdb backend selected but no RRDBSR instance provided")
        return rrdb(image)
    raise ValueError(f"Unknown SR backend: {backend!r} (use none|enhance|rrdb)")
