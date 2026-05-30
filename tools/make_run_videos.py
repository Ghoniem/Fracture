"""Render per-cycle PNG sequences from a simulation run into MP4 videos.

Walks <run_dir>/outer_*_growth/cycle_*/ in sorted order and, for each named
metric, encodes the frames into <run_dir>/<out_subdir>/<metric>.mp4.

The four default metrics are:
    deformed_network
    brazilian_disk_sxx_contour
    brazilian_disk_syy_contour
    brazilian_disk_sxy_contour

A metric with zero frames on disk is skipped with a warning (typical for the
three contour metrics until per-cycle saves are wired into the pipeline).

Usage:
    python tools/make_run_videos.py <run_dir> [--fps 5] [--out-subdir videos]
                                    [--metrics deformed_network ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import imageio.v2 as imageio
from PIL import Image

DEFAULT_METRICS: Sequence[str] = (
    "deformed_network",
    "brazilian_disk_sxx_contour",
    "brazilian_disk_syy_contour",
    "brazilian_disk_sxy_contour",
)


def _cycle_dirs(run_dir: Path) -> List[Path]:
    """Return cycle directories in chronological order across all outer cycles.

    Order: outer_01_growth/cycle_00_initial, outer_01_growth/cycle_01, ...,
           outer_02_growth/cycle_00_initial, ... etc.
    Lexicographic sort gives the right order because of the zero-padded indices
    and 'cycle_00_initial' < 'cycle_01' < 'cycle_02' < 'cycle_NN_terminated_*'.
    """
    outers = sorted(p for p in run_dir.glob("outer_*_growth") if p.is_dir())
    out: List[Path] = []
    for outer in outers:
        out.extend(sorted(c for c in outer.glob("cycle_*") if c.is_dir()))
    return out


def _collect_frames(cycle_dirs: Sequence[Path], filename: str) -> List[Path]:
    return [d / filename for d in cycle_dirs if (d / filename).exists()]


def _pad_to(image: Image.Image, target: Tuple[int, int]) -> Image.Image:
    """Pad image (with white) to target (W, H), preserving original placement."""
    tw, th = target
    if image.size == (tw, th):
        return image
    canvas = Image.new("RGB", (tw, th), (255, 255, 255))
    canvas.paste(image.convert("RGB"), (0, 0))
    return canvas


def _even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def _encode(frames: Sequence[Path], out_path: Path, fps: int) -> None:
    sizes = []
    for p in frames:
        with Image.open(p) as im:
            sizes.append(im.size)
    max_w = _even(max(s[0] for s in sizes))
    max_h = _even(max(s[1] for s in sizes))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for p in frames:
            with Image.open(p) as im:
                padded = _pad_to(im, (max_w, max_h))
                writer.append_data(_pil_to_array(padded))
    finally:
        writer.close()


def _pil_to_array(im: Image.Image):
    import numpy as np
    return np.asarray(im.convert("RGB"))


def make_videos(
    run_dir: Path,
    out_subdir: str = "videos",
    metrics: Sequence[str] = DEFAULT_METRICS,
    fps: int = 5,
) -> List[Path]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist or is not a directory: {run_dir}")

    cycle_dirs = _cycle_dirs(run_dir)
    if not cycle_dirs:
        raise SystemExit(f"no outer_*_growth/cycle_* directories found under {run_dir}")

    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for metric in metrics:
        frames = _collect_frames(cycle_dirs, f"{metric}.png")
        if not frames:
            print(f"[skip] {metric}: 0 frames found")
            continue
        out_path = out_dir / f"{metric}.mp4"
        print(f"[encode] {metric}: {len(frames)} frames @ {fps} fps -> {out_path.name}")
        _encode(frames, out_path, fps=fps)
        written.append(out_path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("run_dir", type=Path, help="path to a run directory (e.g. .../output/20260529_134843_fragmentation)")
    ap.add_argument("--out-subdir", default="videos", help="subdirectory under run_dir for mp4 output (default: videos)")
    ap.add_argument("--fps", type=int, default=5, help="frames per second (default: 5)")
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help=f"metric stem(s) to render (default: {' '.join(DEFAULT_METRICS)})",
    )
    args = ap.parse_args(argv)
    make_videos(args.run_dir, out_subdir=args.out_subdir, metrics=args.metrics, fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
