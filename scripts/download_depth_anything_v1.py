#!/usr/bin/env python3
"""Download official Depth Anything V1 checkpoints into weights/depth-anything/.

Features:
  - Auto-selects a reachable HuggingFace endpoint: tries huggingface.co first,
    then the hf-mirror.com mirror (useful behind the GFW).  Set HF_ENDPOINT in
    the environment to force a specific endpoint.
  - Auto-detects the repo type (space/model) that hosts the weight files.
  - Downloads the requested encoder(s) and verifies each file is non-empty.
  - ``--probe-only`` lists the repo contents without downloading.

Usage:
  python scripts/download_depth_anything_v1.py                       # vitb (default)
  python scripts/download_depth_anything_v1.py --encoders vits vitb   # both
  python scripts/download_depth_anything_v1.py --probe-only           # inspect repo
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WANDB_DISABLED", "true")

EXPECTED_SIZE_MB = {"vits": 96, "vitb": 371, "vitl": 1280}
REPO_CANDIDATES = [
    ("LiheYoung/Depth-Anything", "space"),
    ("LiheYoung/Depth-Anything", "model"),
]


def probe_endpoint() -> str | None:
    """Return a reachable HF base URL, preferring the official endpoint."""
    import urllib.request

    socket.setdefaulttimeout(15)
    for name, url in [("hf", "https://huggingface.co"), ("mirror", "https://hf-mirror.com")]:
        try:
            urllib.request.urlopen(url, timeout=15)
            print(f"[probe] connect {name} ({url}): OK")
            return url
        except Exception as e:
            print(f"[probe] connect {name} ({url}): FAIL {type(e).__name__}: {str(e)[:80]}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--encoders", nargs="+", default=["vitb"], choices=["vits", "vitb", "vitl"])
    parser.add_argument("--output", type=Path, default=Path("weights/depth-anything"))
    parser.add_argument("--probe-only", action="store_true", help="List repo files, do not download")
    args = parser.parse_args()

    output = (PROJECT_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output dir: {output}")

    endpoint = probe_endpoint()
    if endpoint is None:
        print("ERROR: no HuggingFace endpoint is reachable.")
        print("       Set HF_ENDPOINT=https://your-mirror or configure a proxy and retry.")
        sys.exit(2)
    if endpoint != "https://huggingface.co":
        # Must be set BEFORE importing huggingface_hub so its constants pick it up.
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"[setup] using HF_ENDPOINT={endpoint}")

    from huggingface_hub import hf_hub_download, list_repo_files

    # Detect which repo (space/model) hosts the weight files.  Weights may live
    # in a subdirectory (e.g. ``checkpoints/``), so match by suffix not prefix.
    repo = None
    da_files: list[str] = []
    for rid, rtype in REPO_CANDIDATES:
        try:
            files = list_repo_files(rid, repo_type=rtype)
            pth = [f for f in files if f.endswith(".pth")]
            da = [f for f in pth if "depth_anything" in f.lower()]
            print(f"[probe] repo {rid} ({rtype}): {len(files)} files, .pth={pth[:10]}")
            if da:
                print(f"[probe]   depth_anything weights: {da}")
                repo = (rid, rtype)
                da_files = da
                break
        except Exception as e:
            print(f"[probe] repo {rid} ({rtype}): FAIL {type(e).__name__}: {str(e)[:120]}")
    if repo is None:
        print("ERROR: could not locate the Depth Anything V1 weight repo on HuggingFace.")
        sys.exit(3)
    rid, rtype = repo

    if args.probe_only:
        print(f"\n[probe-only] repo {rid} ({rtype}). DA weights found: {da_files}")
        return

    results = []
    for enc in args.encoders:
        target = f"depth_anything_{enc}14.pth"
        # Match the encoder's file, allowing a subdirectory prefix.
        matches = [f for f in da_files if f == target or f.endswith("/" + target)]
        if not matches:
            print(f"[fail] {target} not found in {rid} ({rtype}). Available: {da_files}")
            results.append((enc, str(output / target), 0, "not_found"))
            continue
        remote = matches[0]
        dest = output / target
        if dest.is_file() and dest.stat().st_size > 0:
            size = dest.stat().st_size
            print(f"[skip] {target} already exists (~{size / 2 ** 20:.0f} MB)")
            results.append((enc, str(dest), size, "skipped"))
            continue
        print(f"\n[download] {remote} from {rid} ({rtype}) -> {dest}")
        t0 = time.time()
        try:
            path = hf_hub_download(
                repo_id=rid, filename=remote, repo_type=rtype, local_dir=str(output)
            )
            downloaded = Path(path)
            # If the file landed in a subdir (e.g. checkpoints/), move it top-level.
            if downloaded.name == target and downloaded.parent != output:
                downloaded.replace(dest)
                downloaded = dest
            size = downloaded.stat().st_size
            print(f"[done] {target}: ~{size / 2 ** 20:.0f} MB in {time.time() - t0:.1f}s")
            results.append((enc, str(downloaded), size, "downloaded"))
        except Exception as e:
            print(f"[fail] {target}: {type(e).__name__}: {str(e)[:200]}")
            results.append((enc, str(dest), 0, f"failed:{type(e).__name__}"))

    print("\n=== summary ===")
    for enc, path, size, status in results:
        mb = size / 2 ** 20 if size else 0
        expected = EXPECTED_SIZE_MB.get(enc, 0)
        flag = "" if size else "  <-- MISSING"
        print(f"  {enc:5s}  {status:14s}  {size:>12} bytes (~{mb:>5.0f} MB, expected ~{expected})  {path}{flag}")
    if any(size == 0 for _, _, size, _ in results):
        sys.exit(4)


if __name__ == "__main__":
    main()
