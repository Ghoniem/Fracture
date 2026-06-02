"""Render per-cycle PNG sequences from a simulation run into MP4 videos.

Walks <run_dir>/STEP_NN (preferred) or <run_dir>/outer_*_growth/cycle_*/
(legacy) in sorted order and, for each named metric, encodes the frames into
<run_dir>/<out_subdir>/<metric>.mp4.

Default metrics:
    deformed_network
    network_graph_connectivity
    brazilian_disk_sxx_contour
    brazilian_disk_syy_contour
    brazilian_disk_sxy_contour

For network-style overlays (deformed_network, network_graph_connectivity) the
encoder composites each frame onto a background image (default:
<run_dir>/brazilian_disk_Syy_polar.png) so the polar stress field shows
through the frame's white space. Near-white pixels in the frame are made
transparent before compositing. Pass --no-bg to disable, --bg <path> to
override, or --bg-metrics <names...> to change which metrics get a bg.

A metric with zero frames on disk is skipped with a warning.

Usage:
    python tools/make_run_videos.py <run_dir> [--fps 5] [--out-subdir videos]
                                    [--metrics deformed_network ...]
                                    [--bg PATH | --no-bg]
                                    [--bg-metrics deformed_network ...]
                                    [--bg-threshold 245]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image

DEFAULT_METRICS: Sequence[str] = (
    "deformed_network",
    "network_graph_connectivity",
    "brazilian_disk_sxx_contour",
    "brazilian_disk_syy_contour",
    "brazilian_disk_sxy_contour",
)

DEFAULT_BG_FILENAME: str = "brazilian_disk_Syy_polar.png"
DEFAULT_BG_METRICS: Sequence[str] = (
    "deformed_network",
    "network_graph_connectivity",
)
DEFAULT_BG_THRESHOLD: int = 245


def _cycle_dirs(run_dir: Path) -> List[Path]:
    """Return step / cycle directories in chronological order.

    Prefers the new flat STEP_NN layout (run_dir/STEP_*) emitted by the
    step-based driver. Falls back to the legacy outer_NN_growth/cycle_NN
    layout when no STEP_* dirs are present.
    """
    step_dirs = sorted(p for p in run_dir.glob("STEP_*") if p.is_dir())
    if step_dirs:
        return step_dirs

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


def _disk_bbox_saturated(
    im: Image.Image, sat_thresh: int = 25, val_thresh: int = 50
) -> Optional[Tuple[int, int, int, int]]:
    """Bbox of the largest CC of saturated (colormap-style) pixels (l, t, r, b).

    Suitable for color-filled polar contour plots where the disk is the
    largest such blob; cleanly excludes the colorbar (a smaller separate CC)
    and gray axes/text.
    """
    import scipy.ndimage as ndi
    arr = np.array(im.convert("RGB")).astype(np.int16)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    mask = ((maxc - minc) > sat_thresh) & (maxc > val_thresh)
    if not mask.any():
        return None
    labels, n = ndi.label(mask)
    if n == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    ys, xs = np.where(labels == largest)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _disk_bbox_gray_ring(
    im: Image.Image,
    gray_low: int = 120,
    gray_high: int = 200,
    chroma_max: int = 15,
    dilate_iter: int = 15,
) -> Optional[Tuple[int, int, int, int]]:
    """Bbox of the largest CC of near-gray mid-intensity pixels (l, t, r, b).

    Suitable for network/connectivity frames where the disk is drawn as a
    gray boundary ring on a mostly-white background. Dilation closes small
    gaps so the ring forms a single connected component.
    """
    import scipy.ndimage as ndi
    arr = np.array(im.convert("RGB")).astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    is_gray = (
        (np.abs(r - g) <= chroma_max)
        & (np.abs(g - b) <= chroma_max)
        & (r >= gray_low)
        & (r <= gray_high)
    )
    if not is_gray.any():
        return None
    closed = ndi.binary_dilation(is_gray, iterations=int(dilate_iter))
    labels, n = ndi.label(closed)
    if n == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    ys, xs = np.where(labels == largest)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _frame_keyed(
    frame_path: Path, target: Tuple[int, int], white_threshold: int
) -> Image.Image:
    """Open a frame, pad it onto a transparent canvas of `target` (W, H), and
    knock out near-white pixels (alpha=0) so the bg shows through behind them.
    """
    tw, th = target
    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    with Image.open(frame_path) as im:
        canvas.paste(im.convert("RGBA"), (0, 0))
    arr = np.array(canvas)  # (H, W, 4) uint8
    near_white = (arr[..., :3] >= int(white_threshold)).all(axis=-1)
    arr[..., 3][near_white] = 0
    return Image.fromarray(arr, mode="RGBA")


def _even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def _bbox_square(bbox: Tuple[int, int, int, int], tol: float = 0.05) -> bool:
    """True iff bbox is approximately square (the disk should be)."""
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return False
    return abs(w / h - 1.0) <= tol


def _encode(
    frames: Sequence[Path],
    out_path: Path,
    fps: int,
    bg_path: Optional[Path] = None,
    white_threshold: int = DEFAULT_BG_THRESHOLD,
    bg_scale: float = 1.0,
    bg_alpha: float = 1.0,
) -> None:
    sizes = []
    for p in frames:
        with Image.open(p) as im:
            sizes.append(im.size)
    max_w = _even(max(s[0] for s in sizes))
    max_h = _even(max(s[1] for s in sizes))

    bg_disk_im: Optional[Image.Image] = None
    fallback_bbox: Optional[Tuple[int, int, int, int]] = None
    if bg_path:
        with Image.open(bg_path) as raw_bg:
            bg_rgb = raw_bg.convert("RGB")
            bg_bbox = _disk_bbox_saturated(bg_rgb)
            if bg_bbox is None:
                print(f"[bg] disk detection failed in {bg_path.name}; using full image")
                bg_disk_im = bg_rgb.copy()
            else:
                bg_disk_im = bg_rgb.crop(bg_bbox)
        with Image.open(frames[0]) as f0:
            fallback_bbox = _disk_bbox_gray_ring(f0.convert("RGB"))
        if fallback_bbox is None or not _bbox_square(fallback_bbox):
            fw, fh = sizes[0]
            mx, my = int(0.10 * fw), int(0.06 * fh)
            fallback_bbox = (mx, my, fw - mx, fh - my)
            print(
                f"[bg] first-frame disk detection unreliable; using central fallback {fallback_bbox}"
            )
        else:
            print(
                f"[bg] bg disk crop = {bg_bbox}, first-frame disk = {fallback_bbox}"
            )

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
            if bg_disk_im is not None and fallback_bbox is not None:
                with Image.open(p) as im:
                    rgb = im.convert("RGB")
                bb = _disk_bbox_gray_ring(rgb)
                if bb is None or not _bbox_square(bb):
                    bb = fallback_bbox
                fl, ft, fr, fb = bb
                fdw, fdh = fr - fl, fb - ft
                new_w = max(1, int(round(fdw * float(bg_scale))))
                new_h = max(1, int(round(fdh * float(bg_scale))))
                bg_for_frame = bg_disk_im.resize((new_w, new_h), Image.LANCZOS)
                if float(bg_alpha) < 1.0:
                    a = float(bg_alpha)
                    arr = np.array(bg_for_frame).astype(np.float32)
                    arr = arr * a + 255.0 * (1.0 - a)
                    bg_for_frame = Image.fromarray(
                        np.clip(arr, 0, 255).astype(np.uint8)
                    )
                off_x = fl + (fdw - new_w) // 2
                off_y = ft + (fdh - new_h) // 2
                canvas_bg = Image.new("RGB", (max_w, max_h), (255, 255, 255))
                canvas_bg.paste(bg_for_frame, (off_x, off_y))
                keyed = _frame_keyed(p, (max_w, max_h), white_threshold)
                composite = canvas_bg.convert("RGBA")
                composite.alpha_composite(keyed)
                writer.append_data(_pil_to_array(composite))
            else:
                with Image.open(p) as im:
                    padded = _pad_to(im, (max_w, max_h))
                    writer.append_data(_pil_to_array(padded))
    finally:
        writer.close()


def _pil_to_array(im: Image.Image):
    return np.asarray(im.convert("RGB"))


def make_videos(
    run_dir: Path,
    out_subdir: str = "videos",
    metrics: Sequence[str] = DEFAULT_METRICS,
    fps: int = 5,
    bg_path: Optional[Path] = None,
    bg_metrics: Sequence[str] = DEFAULT_BG_METRICS,
    bg_white_threshold: int = DEFAULT_BG_THRESHOLD,
    bg_scale: float = 1.0,
    bg_alpha: float = 1.0,
    out_suffix: str = "",
) -> List[Path]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist or is not a directory: {run_dir}")

    cycle_dirs = _cycle_dirs(run_dir)
    if not cycle_dirs:
        raise SystemExit(f"no STEP_* or outer_*_growth/cycle_* directories found under {run_dir}")

    bg_metrics_set = set(bg_metrics or ())
    resolved_bg: Optional[Path] = None
    if bg_metrics_set:
        if bg_path is not None:
            resolved_bg = Path(bg_path).resolve()
            if not resolved_bg.is_file():
                print(f"[bg] requested bg not found: {resolved_bg}; disabling overlay")
                resolved_bg = None
        else:
            cand = run_dir / DEFAULT_BG_FILENAME
            if cand.is_file():
                resolved_bg = cand
                print(f"[bg] using {cand.name} as background for {sorted(bg_metrics_set)}")
            else:
                print(f"[bg] no {DEFAULT_BG_FILENAME} in {run_dir}; no overlay")

    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for metric in metrics:
        frames = _collect_frames(cycle_dirs, f"{metric}.png")
        if not frames:
            print(f"[skip] {metric}: 0 frames found")
            continue
        out_path = out_dir / f"{metric}{out_suffix}.mp4"
        bg_for_this = resolved_bg if metric in bg_metrics_set else None
        bg_tag = " (bg)" if bg_for_this is not None else ""
        print(f"[encode] {metric}: {len(frames)} frames @ {fps} fps -> {out_path.name}{bg_tag}")
        _encode(
            frames,
            out_path,
            fps=fps,
            bg_path=bg_for_this,
            white_threshold=bg_white_threshold,
            bg_scale=bg_scale,
            bg_alpha=bg_alpha,
        )
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
    ap.add_argument(
        "--bg",
        type=Path,
        default=None,
        help=f"background image for --bg-metrics videos (default: <run_dir>/{DEFAULT_BG_FILENAME} if present)",
    )
    ap.add_argument(
        "--no-bg",
        action="store_true",
        help="disable background overlay even if a default bg image is present",
    )
    ap.add_argument(
        "--bg-metrics",
        nargs="*",
        default=list(DEFAULT_BG_METRICS),
        help=f"metric stem(s) that get a background overlay (default: {' '.join(DEFAULT_BG_METRICS)})",
    )
    ap.add_argument(
        "--bg-threshold",
        type=int,
        default=DEFAULT_BG_THRESHOLD,
        help=f"RGB component value to treat as near-white and knock out (default: {DEFAULT_BG_THRESHOLD})",
    )
    ap.add_argument(
        "--bg-scale",
        type=float,
        default=1.0,
        help="scale the bg disk relative to the frame disk (default: 1.0; e.g. 0.833 = 5/6)",
    )
    ap.add_argument(
        "--bg-alpha",
        type=float,
        default=1.0,
        help="bg opacity blended against white (0.0=transparent, 1.0=opaque; default: 1.0)",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="suffix appended to each output filename before .mp4 (e.g. '_syy')",
    )
    args = ap.parse_args(argv)
    bg_metrics = [] if args.no_bg else args.bg_metrics
    make_videos(
        args.run_dir,
        out_subdir=args.out_subdir,
        metrics=args.metrics,
        fps=args.fps,
        bg_path=args.bg,
        bg_metrics=bg_metrics,
        bg_white_threshold=args.bg_threshold,
        bg_scale=args.bg_scale,
        bg_alpha=args.bg_alpha,
        out_suffix=args.out_suffix,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
