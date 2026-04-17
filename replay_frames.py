"""
Offline replay for frames captured by crossybot_v2 --log-frames.

Reads a JSONL log of raw (occ_lines, color_lines, t) per frame, feeds them
through a fresh VelocityEstimator using the logged timestamps, and writes a
companion JSONL of detections. Kalman tuning constants can be overridden via
CLI flags so you can sweep values without rerunning the emulator.

Examples:
  # Replay with the visualization window (default)
  python replay_frames.py session.jsonl

  # Headless replay, just write detections
  python replay_frames.py session.jsonl --no-viz --out dets.jsonl

  # Sweep Q_vel and watch the effect
  python replay_frames.py session.jsonl --q-vel 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import crossybot_v2


def _apply_overrides(args: argparse.Namespace) -> None:
    """Monkey-patch module-level Kalman constants before instantiating."""
    overrides = {
        "_KF_Q_POS": args.q_pos,
        "_KF_Q_VEL": args.q_vel,
        "_KF_R": args.r,
        "_KF_GATE_CHI2": args.gate_chi2,
        "_KF_MAX_GATED_FRAMES": args.max_gated,
        "_KF_MERGE_LEN_RATIO": args.merge_len_ratio,
        "_KF_MERGE_OVERLAP_FRAC": args.merge_overlap_frac,
    }
    applied = []
    for name, val in overrides.items():
        if val is not None:
            setattr(crossybot_v2, name, val)
            applied.append(f"{name}={val}")
    if applied:
        print("[replay] overrides: " + ", ".join(applied), file=sys.stderr)


def _clean_det(det: dict) -> dict:
    """Make a detection dict JSON-serialisable (mean_color_bgr is a tuple)."""
    out = dict(det)
    if "mean_color_bgr" in out:
        out["mean_color_bgr"] = list(out["mean_color_bgr"])
    return out


WIN = "replay (space=pause, →/.=next, ←/,=prev-frame [current-frame only], q=quit)"


def _stack_band_vises(
    band_vises: list[tuple[int, np.ndarray]],
    frame_idx: int,
    t: float,
    scale: int,
) -> np.ndarray:
    """Vertically stack per-band vis strips with a band label and header."""
    if not band_vises:
        return np.zeros((40, 240 * scale, 3), np.uint8)
    # Pad all strips to same width
    max_w = max(v.shape[1] for _, v in band_vises)
    rows = []
    for bi, v in band_vises:
        if v.shape[1] < max_w:
            pad = np.zeros((v.shape[0], max_w - v.shape[1], 3), np.uint8)
            v = np.hstack([v, pad])
        label = np.full((v.shape[0], 30, 3), 20, np.uint8)
        cv2.putText(label, f"b{bi}", (2, v.shape[0] // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        rows.append(np.hstack([label, v]))
        rows.append(np.full((2, max_w + 30, 3), 0, np.uint8))  # separator
    stacked = np.vstack(rows) if rows else np.zeros((40, max_w + 30, 3), np.uint8)
    # Header
    header = np.full((24, stacked.shape[1], 3), 30, np.uint8)
    cv2.putText(header, f"frame {frame_idx}  t={t:.3f}s",
                (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    out = np.vstack([header, stacked])
    if scale > 1:
        out = cv2.resize(out, (out.shape[1] * scale, out.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)
    return out


def _run_viz(args, ve, frames: list[dict]) -> tuple[int, int, int]:
    """Interactive playback. Returns (n_frames, n_dets, n_gated)."""
    decode = crossybot_v2.FrameLogger.decode
    n_dets = n_gated = 0
    out_records: list[str] = [] if args.out else []
    idx = 0
    paused = False
    step_once = False

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while 0 <= idx < len(frames):
        rec = frames[idx]
        t = float(rec["t"])
        occ_lines = [decode(b["occ"]) for b in rec["bands"]]
        color_lines = [decode(b["color"]) for b in rec["bands"]]

        # On step-backwards, VelocityEstimator state is stale; rebuild from frame 0.
        # Handled by caller via _rebuild_to(idx). Here we just forward.
        ve.push_occ_lines(occ_lines, t_now=t)
        ve.push_color_lines(color_lines, t_now=t)

        band_vises: list[tuple[int, np.ndarray]] = []
        per_band_dets = []
        for bi in range(len(occ_lines)):
            dets, vis = ve.hash_velocity(
                band=bi, occ_line=occ_lines[bi],
                color_line=color_lines[bi], t_now=t,
            )
            if vis is not None:
                band_vises.append((bi, vis))
            if dets:
                per_band_dets.append({"band": bi, "dets": [_clean_det(d) for d in dets]})
                n_dets += sum(1 for d in dets if not d.get("predicted"))
                n_gated += sum(1 for d in dets if d.get("gated"))

        if args.out:
            out_records.append(json.dumps({"t": t, "bands": per_band_dets}))

        img = _stack_band_vises(band_vises, idx, t, args.scale)
        cv2.imshow(WIN, img)

        # Handle keys
        delay = 0 if paused else max(1, int(args.delay_ms))
        key = cv2.waitKey(delay) & 0xFFFF
        # Common keys: q=113, space=32, ,=44 ('<' unshifted), .=46
        # Arrow keys are platform-dependent under cv2.waitKey — often unreliable;
        # fall back on punctuation for reliable cross-platform control.
        if key == ord("q") or key == 27:
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord(".") or key == ord(">"):
            idx += 1
            paused = True
            continue
        if key == ord(",") or key == ord("<"):
            # Step backwards: we must rebuild state from scratch up to idx-1
            target = max(0, idx - 1)
            ve = crossybot_v2.VelocityEstimator()
            ve._kf_debug = args.debug
            # Replay through frames[0..target-1] silently, but only keep counts if
            # we haven't already seen them (we have, so don't double-count here).
            # Simpler: re-init and re-run without incrementing global counters.
            _silent_rebuild(ve, frames, target, decode, t_override=True)
            idx = target
            paused = True
            continue
        # default: advance if not paused
        if not paused:
            idx += 1

    cv2.destroyWindow(WIN)

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(out_records) + ("\n" if out_records else ""))

    return idx, n_dets, n_gated


def _silent_rebuild(ve, frames, upto: int, decode, t_override: bool) -> None:
    """Run ve through frames[0..upto-1] without yielding visuals."""
    for k in range(upto):
        rec = frames[k]
        t = float(rec["t"])
        occ_lines = [decode(b["occ"]) for b in rec["bands"]]
        color_lines = [decode(b["color"]) for b in rec["bands"]]
        ve.push_occ_lines(occ_lines, t_now=t)
        ve.push_color_lines(color_lines, t_now=t)
        for bi in range(len(occ_lines)):
            ve.hash_velocity(
                band=bi, occ_line=occ_lines[bi],
                color_line=color_lines[bi], t_now=t,
            )


def main() -> int:
    p = argparse.ArgumentParser(description="Replay frame log through VelocityEstimator")
    p.add_argument("input", help="JSONL frame log from --log-frames")
    p.add_argument("--out", default=None, help="JSONL output for per-frame detections")
    p.add_argument("--limit", type=int, default=0, help="Stop after N frames (0 = all)")
    p.add_argument("--debug", action="store_true", help="Enable VelocityEstimator._kf_debug telemetry")
    p.add_argument("--no-viz", action="store_true",
                   help="Disable the interactive visualization window (headless replay)")
    p.add_argument("--delay-ms", type=int, default=33,
                   help="Frame delay in viz mode (default 33ms ≈ 30fps)")
    p.add_argument("--scale", type=int, default=2,
                   help="Integer upscale factor for viz output")
    p.add_argument("--q-pos", type=float, default=None, help="Override _KF_Q_POS")
    p.add_argument("--q-vel", type=float, default=None, help="Override _KF_Q_VEL")
    p.add_argument("--r", type=float, default=None, help="Override _KF_R")
    p.add_argument("--gate-chi2", type=float, default=None, help="Override _KF_GATE_CHI2")
    p.add_argument("--max-gated", type=int, default=None, help="Override _KF_MAX_GATED_FRAMES")
    p.add_argument("--merge-len-ratio", type=float, default=None, help="Override _KF_MERGE_LEN_RATIO")
    p.add_argument("--merge-overlap-frac", type=float, default=None, help="Override _KF_MERGE_OVERLAP_FRAC")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        p.error(f"input not found: {in_path}")

    _apply_overrides(args)

    ve = crossybot_v2.VelocityEstimator()
    ve._kf_debug = args.debug
    decode = crossybot_v2.FrameLogger.decode

    # Load all frames into memory so viz can step backwards
    frames: list[dict] = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
            if args.limit and len(frames) >= args.limit:
                break

    if not args.no_viz:
        n_frames, n_dets, n_gated = _run_viz(args, ve, frames)
    else:
        out_f = open(args.out, "w") if args.out else None
        n_frames = n_dets = n_gated = 0
        for rec in frames:
            t = float(rec["t"])
            occ_lines = [decode(b["occ"]) for b in rec["bands"]]
            color_lines = [decode(b["color"]) for b in rec["bands"]]
            ve.push_occ_lines(occ_lines, t_now=t)
            ve.push_color_lines(color_lines, t_now=t)
            per_band_dets = []
            for bi in range(len(occ_lines)):
                dets, _ = ve.hash_velocity(
                    band=bi, occ_line=occ_lines[bi],
                    color_line=color_lines[bi], t_now=t,
                )
                if dets:
                    per_band_dets.append({
                        "band": bi,
                        "dets": [_clean_det(d) for d in dets],
                    })
                    n_dets += sum(1 for d in dets if not d.get("predicted"))
                    n_gated += sum(1 for d in dets if d.get("gated"))
            if out_f is not None:
                out_f.write(json.dumps({"t": t, "bands": per_band_dets}) + "\n")
            n_frames += 1
        if out_f is not None:
            out_f.close()

    print(f"[replay] frames={n_frames} dets={n_dets} gated={n_gated} "
          f"ratio={n_gated / max(1, n_dets):.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
