from __future__ import annotations
import cv2
import numpy as np
import json
import time
import mss
try:
    import pyautogui
except Exception:
    pyautogui = None
try:
    import pygetwindow as gw
except ImportError:
    gw = None
import matplotlib
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Deque, Sequence, Set
from collections import defaultdict, deque
from enum import Enum
import argparse
import os
import subprocess
import threading
try:
    import Quartz
except ImportError:
    Quartz = None
import abc
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import FuncFormatter

class ImageOps:
    @staticmethod
    def _is_scalar_like(value: Any) -> bool:
        try:
            arr = np.asarray(value)
        except Exception:
            return False
        if arr.ndim == 0:
            return True
        if arr.ndim == 1 and arr.size in (1, 3, 4):
            return True
        return False

    @staticmethod
    def split(src, *, download: bool = True):
        return cv2.split(np.asarray(src))

    @staticmethod
    def in_range(src, lower, upper, *, download: bool = True):
        return cv2.inRange(np.asarray(src), lower, upper)

    @staticmethod
    def bitwise_and(a, b, *, download: bool = True):
        return cv2.bitwise_and(np.asarray(a), np.asarray(b))

    @staticmethod
    def bitwise_or(a, b, *, download: bool = True):
        return cv2.bitwise_or(np.asarray(a), np.asarray(b))

    @staticmethod
    def bitwise_not(src, *, download: bool = True):
        return cv2.bitwise_not(np.asarray(src))

    @staticmethod
    def add(a, b, *, download: bool = True):
        arr_a = np.asarray(a)
        if ImageOps._is_scalar_like(b):
            scalar = np.asarray(b).astype(arr_a.dtype, copy=False)
            return cv2.add(arr_a, np.full_like(arr_a, scalar, dtype=arr_a.dtype))
        return cv2.add(arr_a, np.asarray(b, dtype=arr_a.dtype))

    @staticmethod
    def subtract(a, b, *, download: bool = True):
        arr_a = np.asarray(a)
        if ImageOps._is_scalar_like(b):
            scalar = np.asarray(b).astype(arr_a.dtype, copy=False)
            return cv2.subtract(arr_a, np.full_like(arr_a, scalar, dtype=arr_a.dtype))
        return cv2.subtract(arr_a, np.asarray(b, dtype=arr_a.dtype))

    @staticmethod
    def absdiff(a, b, *, download: bool = True):
        arr_a = np.asarray(a)
        if ImageOps._is_scalar_like(b):
            arr_b = np.full_like(arr_a, fill_value=float(np.asarray(b)))
        else:
            arr_b = np.asarray(b, dtype=arr_a.dtype)
        return cv2.absdiff(arr_a, arr_b)

    @staticmethod
    def threshold(src, thresh, maxval, thresh_type, *, download: bool = True):
        return cv2.threshold(np.asarray(src), thresh, maxval, thresh_type)[1]

    @staticmethod
    def cvt_color(src, code, *, download: bool = True):
        return cv2.cvtColor(np.asarray(src), code)

    @staticmethod
    def resize(src, dsize, fx=None, fy=None, interpolation=cv2.INTER_LINEAR, *, download: bool = True):
        arr = np.asarray(src)
        if dsize is None:
            if fx is None or fy is None:
                raise ValueError("resize requires explicit dsize or both fx and fy")
            return cv2.resize(arr, None, fx=fx, fy=fy, interpolation=interpolation)
        return cv2.resize(arr, dsize, interpolation=interpolation)

    @staticmethod
    def morphology_ex(src, op, kernel, iterations=1, *, download: bool = True):
        src_arr = np.asarray(src)
        return cv2.morphologyEx(src_arr, op, kernel, iterations=max(1, iterations))

    @staticmethod
    def dilate(src, kernel, iterations=1, *, download: bool = True):
        return cv2.dilate(np.asarray(src), kernel, iterations=max(1, iterations))

    @staticmethod
    def erode(src, kernel, iterations=1, *, download: bool = True):
        return cv2.erode(np.asarray(src), kernel, iterations=max(1, iterations))

    @staticmethod
    def median_blur(src, ksize, *, download: bool = True):
        return cv2.medianBlur(np.asarray(src), int(ksize))

TARGET_W, TARGET_H = 240, 480
BAND_COV_SAMPLE_FRAC = 1.0
BAND_AGREEMENT_THRESH = 0.95
OFFSET_SEARCH_RADIUS = 5
VELOCITY_WINDOW_S = 2.0
VELOCITY_FUTURE_PROJECTION_S = 2.0
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cv2.useOptimized()
matplotlib.use("Agg")

def render_lines_visual_base(
    lines: Sequence[Sequence[int]] | np.ndarray | None,
    width: int,
    height: int,
    x_scale: int = 1,
    y_scale: int = 1,
) -> Optional[np.ndarray]:
    if lines is None or len(lines) == 0 or width <= 0 or height <= 0:
        return None
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    for line in lines:
        if line is None or len(line) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in line]
        cv2.line(
            vis,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (255, 255, 255),
            1,
            cv2.LINE_8,
        )
    if x_scale != 1 or y_scale != 1:
        vis = ImageOps.resize(
            vis,
            (
                vis.shape[1] * max(1, int(x_scale)),
                vis.shape[0] * max(1, int(y_scale)),
            ),
            interpolation=cv2.INTER_NEAREST,
        )
    return vis

class LaneType(Enum):
    GREEN = 0
    ROAD = 1
    WATER_LILYPAD = 2
    WATER_PLATFORM = 3
    TRACK = 4
    UNKNOWN = 9

@dataclass
class Lane:
    idx: int
    y0: int
    y1: int
    height: int
    lane_type: LaneType
    mean_rgb: Tuple[float, float, float]
    confidence: float = 0.0

@dataclass
class CaptureConfig:
    monitor: int = 1
    region: Optional[Tuple[int, int, int, int]] = None

class ScreenCapture:
    def __init__(self, config: CaptureConfig):
        self.config = config
        self.sct = mss.mss()
        self._monitor = self._select_monitor()

    def _select_monitor(self):
        mons = self.sct.monitors
        idx = self.config.monitor if 0 <= self.config.monitor < len(mons) else 1
        mon = mons[idx]
        if self.config.region is not None:
            left, top, w, h = self.config.region
            mon = {"left": left, "top": top, "width": w, "height": h}
        return mon

    def next_frame(self) -> np.ndarray:
        img = np.array(self.sct.grab(self._monitor))
        return img[:, :, :3]

class CaptureBackend(abc.ABC):
    @abc.abstractmethod
    def next_frame(self) -> np.ndarray:
        """Return a BGR frame as a numpy array."""

class InputBackend(abc.ABC):
    @abc.abstractmethod
    def press(self, key: str) -> None:
        """Send a directional key press (up/down/left/right)."""

    @abc.abstractmethod
    def tap(self, x: int, y: int) -> None:
        """Tap at screen coordinates."""


class DesktopCaptureBackend(CaptureBackend):
    def __init__(self, config: CaptureConfig):
        self._cap = ScreenCapture(config)

    def next_frame(self) -> np.ndarray:
        return self._cap.next_frame()


class DesktopInputBackend(InputBackend):
    def press(self, key: str) -> None:
        pyautogui.press(key)

    def tap(self, x: int, y: int) -> None:
        pyautogui.click(x, y)


class AdbCaptureBackend(CaptureBackend):
    def __init__(self, serial: Optional[str] = None,
                 crop: Optional[Tuple[int, int, int, int]] = None):
        self._serial = serial
        self._adb_prefix = ["adb"]
        if serial:
            self._adb_prefix = ["adb", "-s", serial]
        self._crop = crop  # (x, y, w, h) or None

    def next_frame(self) -> np.ndarray:
        cmd = self._adb_prefix + ["exec-out", "screencap", "-p"]
        result = subprocess.run(cmd, capture_output=True, timeout=5)
        if result.returncode != 0:
            raise RuntimeError(f"adb screencap failed: {result.stderr.decode(errors='replace')}")
        png_bytes = result.stdout
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode screencap PNG from adb")
        if self._crop is not None:
            x, y, w, h = self._crop
            ih, iw = img.shape[:2]
            x, y = max(0, x), max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)
            if w > 0 and h > 0:
                img = img[y:y+h, x:x+w]
        return img


class PipeCaptureBackend(CaptureBackend):
    """
    Streams frames from the device via screenrecord H264 → OpenCV VideoCapture.
    A background thread reads continuously; next_frame() returns the latest.
    Designed for Docker/Linux. Achieves 60–90 FPS at 270x480.
    """
    _FIFO_PATH = "/tmp/crossybot_screen.h264"

    def __init__(self, serial: Optional[str] = None,
                 crop: Optional[Tuple[int, int, int, int]] = None):
        self._serial = serial
        self._crop = crop
        self._adb_prefix = ["adb"] + (["-s", serial] if serial else [])
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._running = True
        self._start_pipeline()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _start_pipeline(self) -> None:
        fifo = self._FIFO_PATH
        if os.path.exists(fifo):
            os.unlink(fifo)
        os.mkfifo(fifo)

        # screenrecord writes raw H264 to the FIFO
        def _writer():
            with open(fifo, "wb") as f:
                self._adb_proc = subprocess.Popen(
                    self._adb_prefix + [
                        "exec-out", "screenrecord", "--output-format=h264", "/dev/stdout",
                    ],
                    stdout=f, stderr=subprocess.DEVNULL,
                )
                self._adb_proc.wait()
        self._writer_thread = threading.Thread(target=_writer, daemon=True)
        self._writer_thread.start()

        # OpenCV reads from the FIFO using its built-in ffmpeg decoder
        self._cap = cv2.VideoCapture(fifo)

    def _reader_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                # screenrecord hit 3-min limit — restart
                self._cap.release()
                self._start_pipeline()
                continue
            with self._frame_lock:
                self._latest_frame = frame
            self._frame_ready.set()

    def next_frame(self) -> np.ndarray:
        self._frame_ready.wait()  # only blocks until first frame arrives
        with self._frame_lock:
            img = self._latest_frame
        if self._crop is not None:
            x, y, w, h = self._crop
            img = img[y:y+h, x:x+w]
        return img

    def __del__(self) -> None:
        self._running = False
        if hasattr(self, "_cap"):
            self._cap.release()
        proc = getattr(self, "_adb_proc", None)
        if proc and proc.poll() is None:
            proc.terminate()
        fifo = self._FIFO_PATH
        if os.path.exists(fifo):
            os.unlink(fifo)


class AdbInputBackend(InputBackend):
    _FALLBACK_W = 1080
    _FALLBACK_H = 1920

    def __init__(self, serial: Optional[str] = None):
        self._serial = serial
        self._adb_prefix = ["adb"]
        if serial:
            self._adb_prefix = ["adb", "-s", serial]
        self._dev_w, self._dev_h = self._query_screen_size()
        self._children: List[subprocess.Popen] = []

    def _query_screen_size(self) -> Tuple[int, int]:
        try:
            cmd = self._adb_prefix + ["shell", "wm", "size"]
            result = subprocess.run(cmd, capture_output=True, timeout=5, text=True)
            for line in result.stdout.strip().splitlines():
                if "x" in line:
                    parts = line.split()[-1]  # e.g. "1080x1920"
                    w, h = parts.split("x")
                    return int(w), int(h)
        except Exception:
            pass
        return self._FALLBACK_W, self._FALLBACK_H

    def _rel(self, frac_x: float, frac_y: float) -> Tuple[int, int]:
        return int(frac_x * self._dev_w), int(frac_y * self._dev_h)

    def _build_key_map(self) -> dict:
        cx, cy = self._rel(0.5, 0.42)
        _, dy = self._rel(0.0, 0.10)
        dx, _ = self._rel(0.19, 0.0)
        return {
            "up": f"swipe {cx} {cy} {cx} {cy - dy} 80",
            "down": f"swipe {cx} {cy} {cx} {cy + dy} 80",
            "left": f"swipe {cx} {cy} {cx - dx} {cy} 80",
            "right": f"swipe {cx} {cy} {cx + dx} {cy} 80",
        }

    def _reap_children(self) -> None:
        alive = []
        for p in self._children:
            if p.poll() is None:
                alive.append(p)
        self._children = alive

    def _run(self, shell_cmd: str) -> None:
        self._reap_children()
        cmd = self._adb_prefix + ["shell", "input"] + shell_cmd.split()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._children.append(proc)

    def press(self, key: str) -> None:
        key_map = self._build_key_map()
        shell_cmd = key_map.get(key)
        if shell_cmd is None:
            shell_cmd = f"keyevent {key}"
        self._run(shell_cmd)

    def tap(self, x: int, y: int) -> None:
        self._run(f"tap {x} {y}")


class Visualizer:
    @staticmethod
    def draw_hud(frame_bgr, stats: Dict[str, float], origin=(10, 20)):
        x, y = origin
        for k, v in stats.items():
            if isinstance(v, float):
                text = f"{k}: {v:.1f}"
            else:
                text = f"{k}: {v}"
            cv2.putText(
                frame_bgr,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 22
        return frame_bgr

    @staticmethod
    def draw_lane_bands(frame_bgr, lanes: List[Lane], show_labels: bool = True):
        h, w = frame_bgr.shape[:2]
        for lane in lanes:
            color = (0, 200, 255)
            y0, y1 = int(lane.y0), int(lane.y1)
            cv2.rectangle(frame_bgr, (0, y0), (w - 1, y1), color, 1)
            if show_labels:
                label = lane.lane_type.name
                cv2.putText(
                    frame_bgr,
                    f"{lane.idx}:{label}",
                    (5, max(14, y0 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        return frame_bgr

    @staticmethod
    def make_occ_lines(
        occ_lines: list[np.ndarray],
        band_idx: Optional[int] = None,
        *,
        col_scale: int = 8,
        band_h: int = 18,
        invert: bool = False,
        show_labels: bool = True,
        label_w: int = 80,
        row_sep_px: int = 1,
        row_sep_color: tuple[int, int, int] = (50, 50, 50),
    ) -> np.ndarray:
        def _render_row(idx: int) -> Optional[np.ndarray]:
            line_ = occ_lines[idx]
            if line_ is None:
                return None
            line = (line_.astype(np.uint8) & 1).reshape(1, -1)
            W = line.shape[1]
            vis = (
                (line * 255).astype(np.uint8)
                if invert
                else np.where(line == 0, 255, 0).astype(np.uint8)
            )
            vis = ImageOps.resize(
                vis, (W * col_scale, 1), interpolation=cv2.INTER_NEAREST
            )
            vis = np.repeat(vis, band_h, axis=0)
            row = ImageOps.cvt_color(vis, cv2.COLOR_GRAY2BGR)
            if show_labels:
                lab = np.full((band_h, label_w, 3), (24, 24, 24), np.uint8)
                cv2.putText(
                    lab,
                    f"band {idx}",
                    (6, band_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                row = cv2.hconcat([lab, row])
            return row

        if not occ_lines:
            w = label_w + col_scale
            return np.zeros((band_h, max(1, w), 3), np.uint8)
        if band_idx is not None and band_idx != -1:
            if not (0 <= band_idx < len(occ_lines)) or occ_lines[band_idx] is None:
                w = label_w + col_scale
                return np.zeros((band_h, max(1, w), 3), np.uint8)
            return _render_row(band_idx)
        rows: list[np.ndarray] = []
        for i in range(6, min(20, len(occ_lines))):
            r = _render_row(i)
            if r is not None:
                rows.append(r)
        if not rows:
            w = label_w + col_scale
            return np.zeros((band_h, max(1, w), 3), np.uint8)
        if row_sep_px > 0:
            width = rows[0].shape[1]
            sep = np.full((row_sep_px, width, 3), row_sep_color, np.uint8)
            out = rows[0]
            for r in rows[1:]:
                out = cv2.vconcat([out, sep, r])
            return out
        else:
            return cv2.vconcat(rows)

class DensityMaskUtils:
    @staticmethod
    def fill_small_gaps_1d(mask: np.ndarray, max_gap: int) -> np.ndarray:
        m = mask.astype(bool).copy()
        W = m.shape[0]
        i = 0
        while i < W:
            if m[i]:
                i += 1
                continue
            j = i
            while j < W and not m[j]:
                j += 1
            left_true = (i - 1) >= 0 and m[i - 1]
            right_true = (j < W) and m[j] if j < W else False
            if left_true and right_true and (j - i) <= max_gap:
                m[i:j] = True
            i = j
        return m

    @staticmethod
    def _box_density(mask01: np.ndarray, ksize: int) -> np.ndarray:
        mask = mask01.astype(np.float32, copy=False)
        if mask.max() > 1.0:
            mask = mask * (1.0 / 255.0)
        return mask

    def _ensure_bands(self, n: int) -> None:
        grow = n - len(self._rows)
        if grow <= 0:
            return
        self._rows += [deque() for _ in range(grow)]
        self._times += [deque() for _ in range(grow)]
        self._widths += [0 for _ in range(grow)]
        self._inc_last_x += [0 for _ in range(grow)]
        self._inc_pts_x += [[] for _ in range(grow)]
        self._inc_pts_y += [[] for _ in range(grow)]
        self._inc_oldest_time += [float("nan") for _ in range(grow)]
        self._prob_projection_gradient_history += [{} for _ in range(grow)]
        self._latest_segment_records += [None for _ in range(grow)]
        self._recent_velocity_circles += [None for _ in range(grow)]

    def _valid_band(self, b: int) -> bool:
        return 0 <= b < len(self._rows)

    @staticmethod
    def _coerce_binary_1d(line: np.ndarray) -> np.ndarray:
        li = np.asarray(line)
        if li.ndim != 1:
            li = li.reshape(-1)
        if li.dtype != np.uint8:
            li = li.astype(np.uint8)
        if li.size and li.max() > 1:
            li = (li > 0).astype(np.uint8)
        return li

    def _trim_band(self, b: int) -> None:
        if len(self._times[b]) == 0:
            return
        t_newest = float(self._times[b][-1])
        while (
            len(self._times[b]) > 1
            and (t_newest - float(self._times[b][0])) > self.window_s
        ):
            self._times[b].popleft()
            self._rows[b].popleft()


    @staticmethod
    def _lab_chroma(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        da = a.astype(np.int16) - 128
        db = b.astype(np.int16) - 128
        return cv2.sqrt((da * da + db * db).astype(np.float32))

    @staticmethod
    def gray_density_map_hsv_from(hsv, lab, s_max=80, v_min=80, v_max=130, ksize=11):
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h_gate = ImageOps.in_range(H, 95, 135)
        s_gate = ImageOps.in_range(S, 0, s_max)
        v_gate = ImageOps.in_range(V, v_min, v_max)
        hsv_gate = ImageOps.bitwise_and(h_gate, ImageOps.bitwise_and(s_gate, v_gate))
        aL = lab[..., 1].astype(np.float32)
        bL = lab[..., 2].astype(np.float32)
        da = ImageOps.absdiff(aL, 128.0)
        db = ImageOps.absdiff(bL, 128.0)
        chroma = cv2.sqrt((aL - 128.0) * (aL - 128.0) + (bL - 128.0) * (bL - 128.0))
        lab_gate = ((da <= 10.0) & (db <= 12.0) & (chroma <= 16.0)).astype(np.uint8) * 255
        gray_255 = ImageOps.bitwise_and(hsv_gate, lab_gate)
        gray_255 = ImageOps.morphology_ex(gray_255, cv2.MORPH_OPEN, K3, iterations=1)
        gray_255 = ImageOps.morphology_ex(gray_255, cv2.MORPH_CLOSE, K3, iterations=1)
        gray01 = (gray_255 > 0).astype(np.uint8)
        den = DensityMaskUtils._box_density(gray01, ksize)
        return gray01, den

    @staticmethod
    def blue_density_map_hsv_from(
        hsv, lab, h_lo=95, h_hi=110, s_min=100, v_min=200, ksize=11
    ):
        blue_hsv = ImageOps.in_range(hsv, (h_lo, s_min, v_min), (h_hi, 255, 255))
        bL = lab[..., 2]
        lab_gate = (bL <= 125).astype(np.uint8) * 255
        blue_255 = ImageOps.bitwise_and(blue_hsv, lab_gate)
        blue01 = (blue_255 > 0).astype(np.uint8)
        den = DensityMaskUtils._box_density(blue01, ksize)
        return blue01, den

    @staticmethod
    def green_density_map_hsv_from(hsv, lab, ranges=None, ksize=11):
        if ranges is None:
            r1 = ((38, 100, 205), (46, 200, 255))
            r2 = ((36, 90, 185), (46, 200, 240))
            ranges = [r1, r2]
        mask_hsv = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask_hsv |= ImageOps.in_range(hsv, lo, hi)
        aL, bL = lab[..., 1], lab[..., 2]
        chroma = DensityMaskUtils._lab_chroma(aL, bL)
        lab_gate = ((aL <= 120) & (bL >= 170) & (chroma >= 16.0)).astype(np.uint8) * 255
        water_band = ImageOps.in_range(hsv, (90, 40, 0), (110, 255, 255))
        not_water = ImageOps.bitwise_not(water_band)
        green_255 = ImageOps.bitwise_and(
            ImageOps.bitwise_and(mask_hsv, lab_gate), not_water
        )
        green_255 = ImageOps.morphology_ex(green_255, cv2.MORPH_OPEN, K3, iterations=1)
        green_255 = ImageOps.morphology_ex(green_255, cv2.MORPH_CLOSE, K3, iterations=1)
        green01 = (green_255 > 0).astype(np.uint8)
        den = DensityMaskUtils._box_density(green01, ksize)
        return green01, den

    @staticmethod
    def gray_density_map_hsv(
        bgr: np.ndarray,
        s_max: int = 80,
        v_min: int = 80,
        v_max: int = 130,
        ksize: int = 11,
    ) -> tuple[np.ndarray, np.ndarray]:
        hsv = ImageOps.cvt_color(bgr, cv2.COLOR_BGR2HSV)
        lab = ImageOps.cvt_color(bgr, cv2.COLOR_BGR2Lab)
        return DensityMaskUtils.gray_density_map_hsv_from(
            hsv, lab, s_max, v_min, v_max, ksize
        )

    @staticmethod
    def blue_density_map_hsv(
        bgr: np.ndarray,
        h_lo: int = 96,
        h_hi: int = 101,
        s_min: int = 105,
        v_min: int = 240,
        ksize: int = 11,
    ) -> tuple[np.ndarray, np.ndarray]:
        hsv = ImageOps.cvt_color(bgr, cv2.COLOR_BGR2HSV)
        lab = ImageOps.cvt_color(bgr, cv2.COLOR_BGR2Lab)
        return DensityMaskUtils.blue_density_map_hsv_from(
            hsv, lab, h_lo, h_hi, s_min, v_min, ksize
        )

    @staticmethod
    def free_mask_gray(
        disp_bgr: np.ndarray,
        bands: list[tuple[int, int]],
        lane_types: list['LaneType'],
        *,
        gray_kernel: int = 11,
        gray_tau: float = 0.55,
        blue_kernel: int = 11,
        nonblue_tau: float = 0.55,
        green_kernel: int = 11,
        green_tau: float = 0.55,
        min_vert_frac: float = 0.75,
        consider_bottom_frac: float = 0.75,
        gap_fill_px: int | None = None,
        maps: dict | None = None,
        erode_px: int = 0,
    ) -> np.ndarray:
        H, W = disp_bgr.shape[:2]
        mask = np.ones((H, W), np.uint8)
        if maps is None:
            _, gray_den = DensityMaskUtils.gray_density_map_hsv(
                disp_bgr, ksize=gray_kernel
            )
            _, blue_den = DensityMaskUtils.blue_density_map_hsv(
                disp_bgr, ksize=blue_kernel
            )
            nonblue_den = 1.0 - blue_den
            _, green_den = DensityMaskUtils.green_density_map_hsv(
                disp_bgr, ksize=green_kernel
            )
        else:
            gray_den = maps['gray_den']
            blue_den = maps['blue_den']
            nonblue_den = 1.0 - blue_den
            green_den = maps['green_den']
        for (y0, y1), lt in zip(bands, lane_types):
            if y1 <= y0:
                continue
            band_h = y1 - y0
            h_bot = max(1, int(round(consider_bottom_frac * band_h)))
            ys, ye = y1 - h_bot, y1
            if lt == LaneType.ROAD:
                den_bot = gray_den[ys:ye, :]
                free_bot = den_bot >= gray_tau
            elif lt in (LaneType.WATER_PLATFORM, LaneType.WATER_LILYPAD, LaneType.UNKNOWN):
                den_bot = nonblue_den[ys:ye, :]
                free_bot = den_bot >= nonblue_tau
            elif lt == LaneType.GREEN:
                den_bot = green_den[ys:ye, :]
                free_bot = den_bot >= green_tau
            else:
                continue
            need = int(min_vert_frac * h_bot)
            col_ok = free_bot.sum(axis=0) >= need
            if gap_fill_px is not None and gap_fill_px > 0:
                col_ok = DensityMaskUtils.fill_small_gaps_1d(
                    col_ok, gap_fill_px + (10 if lt == LaneType.ROAD else 0)
                )
            band_mask = np.ones((band_h, W), np.uint8)
            if col_ok.any():
                band_mask[-h_bot:, col_ok] = 0
            mask[y0:y1, :] = band_mask
            if erode_px > 0:
                free01 = (mask == 0).astype(np.uint8)
                k = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (erode_px * 2 + 1, 1)
                )
                free_eroded = cv2.erode(free01, k, iterations=1)
                mask = np.where(free_eroded > 0, 0, 1).astype(np.uint8)
        return mask

class CharacterTools:
    @staticmethod
    def _stack_tiles(tiles, scale=1.0, pad=6, bg=(24, 24, 24)):
        if not isinstance(tiles[0], (list, tuple)):
            tiles = [tiles]

        def to_bgr_u8(im):
            im = np.asarray(im)
            if im.ndim == 2:
                im = ImageOps.cvt_color(im, cv2.COLOR_GRAY2BGR)
            elif im.ndim == 3 and im.shape[2] == 4:
                im = ImageOps.cvt_color(im, cv2.COLOR_BGRA2BGR)
            if im.dtype != np.uint8:
                im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return im

        cell_h = max(to_bgr_u8(img).shape[0] for row in tiles for img in row)
        cell_w = max(to_bgr_u8(img).shape[1] for row in tiles for img in row)
        cols = max(len(row) for row in tiles)
        rows_bgr = []
        for row in tiles:
            row_imgs = []
            for j in range(cols):
                if j < len(row):
                    im = to_bgr_u8(row[j])
                else:
                    im = np.full((1, 1, 3), bg, np.uint8)
                dh, dw = cell_h - im.shape[0], cell_w - im.shape[1]
                im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=bg)
                im = cv2.copyMakeBorder(
                    im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg
                )
                row_imgs.append(im)
            rows_bgr.append(cv2.hconcat(row_imgs))
        grid = cv2.vconcat(rows_bgr)
        if scale != 1.0:
            grid = ImageOps.resize(
                grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )
        return grid

    @staticmethod
    def _label_img_from_cc(labels: np.ndarray, best_label: int) -> np.ndarray:
        h, w = labels.shape
        rng = np.random.RandomState(42)
        max_lab = labels.max()
        colors = np.zeros((max_lab + 1, 3), np.uint8)
        for i in range(1, max_lab + 1):
            colors[i] = rng.randint(40, 220, size=3)
        lbl_bgr = colors[labels]
        if best_label > 0:
            chosen = (labels == best_label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                chosen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(lbl_bgr, contours, -1, (255, 255, 255), 2)
        return lbl_bgr

    @staticmethod
    def detect_character(
        bgr: np.ndarray,
        *,
        bgr_target: Tuple[int, int, int] = (92, 172, 255),
        tol: int = 4,
        min_area_frac: float = 0.0005,
        max_area_frac: float = 0.05,
        open_ksz: int = 3,
        close_ksz: int = 5,
        debug: bool = False,
    ) -> Tuple[
        Optional[Tuple[int, int]],
        Optional[Tuple[int, int, int, int]],
        float,
        Optional[Dict[str, Any]],
    ]:
        H, W = bgr.shape[:2]
        B0, G0, R0 = map(int, bgr_target)
        lo = (max(0, B0 - tol), max(0, G0 - tol), max(0, R0 - tol))
        hi = (min(255, B0 + tol), min(255, G0 + tol), min(255, R0 + tol))
        mask_raw = ImageOps.in_range(bgr, lo, hi)
        n_on = int(cv2.countNonZero(mask_raw))
        K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        K53 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        K35 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        path = "orig"
        if n_on <= 12:
            path = "grow_close_strong"
            mask_boost = ImageOps.dilate(mask_raw, K3, iterations=2)
            mask_boost = ImageOps.morphology_ex(
                mask_boost, cv2.MORPH_CLOSE, K35, iterations=1
            )
            mask_boost = ImageOps.median_blur(mask_boost, 3)
            min_area_abs = 2
        elif n_on <= 80:
            path = "close_dilate_light"
            tmp = ImageOps.morphology_ex(mask_raw, cv2.MORPH_CLOSE, K3, iterations=1)
            mask_boost = ImageOps.dilate(tmp, K3, iterations=1)
            min_area_abs = 8
        else:
            path = "open_close_knobs"
            mask_open = mask_raw
            if open_ksz > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksz, open_ksz))
                mask_open = ImageOps.morphology_ex(
                    mask_open, cv2.MORPH_OPEN, k, iterations=1
                )
            mask_boost = mask_open
            if close_ksz > 0:
                k2 = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (close_ksz, close_ksz)
                )
                mask_boost = ImageOps.morphology_ex(
                    mask_boost, cv2.MORPH_CLOSE, k2, iterations=1
                )
            min_area_abs = int(max(1, min_area_frac * (H * W)))
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_boost, connectivity=8
        )
        frame_area = H * W
        max_area_abs = int(max_area_frac * frame_area)
        best_i, best_area = -1, -1
        for i in range(1, num):
            x, y, w, h, a = stats[i]
            if a < min_area_abs or a > max_area_abs:
                continue
            if w <= 1 and h <= 1:
                continue
            if a > best_area:
                best_i, best_area = i, a
        center = None
        bbox = None
        score = 0.0
        orig_center = None
        if best_i > 0:
            x, y, w, h, a = stats[best_i]
            cx, cy = map(int, centroids[best_i])
            score = float(cv2.countNonZero(mask_raw[y : y + h, x : x + w])) / float(
                max(1, w * h)
            )
            cy_shifted = int(np.clip(cy + 11, 0, H - 1))
            orig_center = (cx, cy)
            center = (cx, cy_shifted)
            bbox = (x, y, w, h)
        else:
            if n_on > 0:
                ys, xs = np.where(mask_raw > 0)
                cx = int(np.clip(round(xs.mean()), 0, W - 1))
                cy = int(np.clip(round(ys.mean()), 0, H - 1))
                pad = 6
                x0 = int(max(0, cx - pad))
                y0 = int(max(0, cy - pad))
                x1 = int(min(W - 1, cx + pad))
                y1 = int(min(H - 1, cy + pad))
                w = x1 - x0 + 1
                h = y1 - y0 + 1
                score = float(
                    cv2.countNonZero(mask_raw[y0 : y1 + 1, x0 : x1 + 1])
                ) / float(max(1, w * h))
                cy_shifted = int(np.clip(cy + 11, 0, H - 1))
                orig_center = (cx, cy)
                center = (cx, cy_shifted)
                bbox = (x0, y0, w, h)
        if not debug:
            return center, bbox, score, None
        labels_vis = CharacterTools._label_img_from_cc(
            labels if "labels" in locals() else np.zeros_like(mask_raw),
            best_i if best_i > 0 else -1,
        )
        overlay = bgr.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if center is not None:
                cv2.circle(overlay, center, 3, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                f"score={score:.2f}",
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            if orig_center is not None:
                cv2.circle(overlay, orig_center, 2, (255, 255, 0), -1)

        def _with_title(img, title):
            im = img.copy()
            cv2.putText(
                im,
                title,
                (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            return im

        target_chip = np.full((40, 120, 3), (B0, G0, R0), np.uint8)
        cv2.putText(
            target_chip,
            f"B:{B0} G:{G0} R:{R0} tol:{tol}",
            (4, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        montage = CharacterTools._stack_tiles(
            [
                [
                    _with_title(bgr, "BGR frame"),
                    _with_title(target_chip, "target"),
                    _with_title(mask_raw, f"mask_raw n={n_on}"),
                ],
                [
                    _with_title(mask_boost, f"mask_boost [{path}]"),
                    _with_title(labels_vis, "CC labels"),
                    np.zeros_like(bgr),
                ],
                [_with_title(overlay, "overlay")],
            ],
            scale=1.0,
        )
        debug_dict = {
            "mask_raw": mask_raw,
            "mask_boost": mask_boost,
            "labels_vis": labels_vis,
            "overlay": overlay,
            "montage": montage,
            "best_label": best_i,
            "stats": stats if "stats" in locals() else None,
            "centroids": centroids if "centroids" in locals() else None,
            "center_unshifted": orig_center,
            "center_shifted": center,
            "bgr_target": bgr_target,
            "tol": tol,
            "lower_bgr": lo,
            "upper_bgr": hi,
            "n_on": n_on,
            "post_path": path,
            "min_area_abs": int(min_area_abs),
            "max_area_abs": int(max_area_abs),
        }
        return center, bbox, score, debug_dict

class GeometryAnalyzer:
    _OFFSET_ROWS_CACHE: dict[tuple[int, int, int, int], list[list[tuple[int, ...]]]] = {}

    @staticmethod
    def _get_offset_band_rows(H: int, lane_h: int, margin_px: int, n_lines: int) -> list[list[tuple[int, ...]]]:
        key = (int(H), int(lane_h), int(margin_px), int(n_lines))
        cache = GeometryAnalyzer._OFFSET_ROWS_CACHE
        if key in cache:
            return cache[key]
        total_offsets = max(1, int(lane_h))
        offsets: list[list[tuple[int, ...]]] = []
        for offset in range(total_offsets):
            bands = GeometryAnalyzer.find_uniform_bands_with_offset(
                int(H), int(lane_h), offset, margin_px=margin_px
            )
            band_rows: list[tuple[int, ...]] = []
            for y0, y1 in bands:
                rows = BandClassifier._pick_band_rows(
                    y0, y1, margin_px=margin_px, n_lines=n_lines
                )
                band_rows.append(tuple(int(r) for r in rows))
            offsets.append(band_rows)
        cache[key] = offsets
        return offsets

    @staticmethod
    def deskew(frame_bgr, M=None, out_size=None):
        if M is None:
            return frame_bgr
        h_out, w_out = out_size
        return cv2.warpAffine(frame_bgr, M, (w_out, h_out), flags=cv2.INTER_LINEAR)

    @staticmethod
    def find_uniform_bands_with_offset(H: int, h: int, o: int, margin_px: int = 2):
        bands = []
        y = o
        while y + h < H:
            y0 = max(0, y + margin_px)
            y1 = min(H, y + h - margin_px)
            bands.append((y0, y1))
            y += h
        return bands

    @staticmethod
    def classify_line_by_masks_fast(
        y: int,
        maps: dict,
        *,
        margin_px: int = 0,
        tau_g: float = 0.55,
        tau_w: float = 0.55,
        tau_r: float = 0.55,
        min_cov: float = 0.18,
        debug: bool = False,
    ) -> LaneType | tuple[LaneType, dict]:
        H, W = maps["gray_den"].shape[:2]
        y = int(np.clip(y, 0, H - 1))
        xs = slice(margin_px, -margin_px or None)
        gden = maps["green_den"][y, xs]
        bden = maps["blue_den"][y, xs]
        rden = maps["gray_den"][y, xs]
        cov_g = float((gden >= tau_g).mean()) if gden.size else 0.0
        cov_w = float((bden >= tau_w).mean()) if bden.size else 0.0
        cov_r = float((rden >= tau_r).mean()) if rden.size else 0.0
        covs = [
            (cov_g, LaneType.GREEN),
            (cov_w, LaneType.WATER_PLATFORM),
            (cov_r, LaneType.ROAD),
        ]
        covs.sort(key=lambda t: t[0], reverse=True)
        label = covs[0][1] if covs[0][0] >= min_cov else LaneType.UNKNOWN
        if not debug:
            return label
        return label, {
            "cov_g": cov_g,
            "cov_w": cov_w,
            "cov_r": cov_r,
            "min_cov": min_cov,
            "y": y,
        }

    @staticmethod
    def classify_band_top_bottom_lines(
        band: tuple[int, int],
        maps: dict,
        *,
        margin_px: int = 0,
        tau_g: float = 0.55,
        tau_w: float = 0.55,
        tau_r: float = 0.55,
        min_cov: float = 0.18,
        debug: bool = False,
    ):
        y0, y1 = int(band[0]), int(band[1])
        if y1 <= y0:
            if debug:
                return {
                    "top": (LaneType.UNKNOWN, {"reason": "empty band"}),
                    "bottom": (LaneType.UNKNOWN, {"reason": "empty band"}),
                }
            return {"top": LaneType.UNKNOWN, "bottom": LaneType.UNKNOWN}
        y_top = int(np.clip(y0 + margin_px, y0, y1 - 1))
        y_bot = int(np.clip(y1 - 1 - margin_px, y0, y1 - 1))
        args = dict(
            margin_px=margin_px,
            tau_g=tau_g,
            tau_w=tau_w,
            tau_r=tau_r,
            min_cov=min_cov,
            debug=debug,
        )
        top_res = GeometryAnalyzer.classify_line_by_masks_fast(y_top, maps, **args)
        bot_res = GeometryAnalyzer.classify_line_by_masks_fast(y_bot, maps, **args)
        return {"top": top_res, "bottom": bot_res}

    @staticmethod
    def band_index_for_y_lower(
        y: int, lane_h: int, offset: int, frame_h: int, n_bands: int
    ) -> Optional[int]:
        if y < offset:
            return None
        k = int(np.floor((y - offset) / float(lane_h)))
        if 0 <= k < n_bands:
            if offset + (k + 1) * lane_h <= frame_h:
                return k
        return None

    @staticmethod
    def build_occ_lines_for_bands(
        free_mask: np.ndarray,
        bands: list[tuple[int, int]],
        lane_h: int,
        use_bottom_slice: bool = True,
        *,
        v_median_ksize: int = 1,
        pad_x: int = 0,
    ) -> list[np.ndarray]:
        H, W = free_mask.shape[:2]
        m = free_mask.astype(np.uint8) & 1
        occ_lines: list[np.ndarray] = []
        for y0, y1 in bands:
            y0c = max(0, min(H, int(y0)))
            y1c = max(0, min(H, int(y1)))
            band_h = y1c - y0c
            if band_h <= 0:
                occ_lines.append(np.ones((W,), np.uint8))
                continue
            if use_bottom_slice:
                slice_h = max(1, min(lane_h, band_h))
                y_top, y_bot = y1c - slice_h, y1c
                y_mid = (y_top + y_bot - 1) // 2
            else:
                y_mid = (y0c + y1c - 1) // 2
            if v_median_ksize > 1:
                r = v_median_ksize // 2
                y_lo = max(0, y_mid - r)
                y_hi = min(H, y_mid + r + 1)
                row = np.median(m[y_lo:y_hi, :], axis=0).astype(np.uint8)
            else:
                row = m[y_mid, :]
            if pad_x > 0:
                k = 2 * int(pad_x) + 1
                row = ImageOps.dilate(
                    row.reshape(1, -1), np.ones((1, k), np.uint8), iterations=1
                ).reshape(-1)
            occ_lines.append(row.astype(np.uint8))
        return occ_lines

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < eps or nb < eps:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def best_offset_by_line_similarity(
        *,
        maps: dict,
        frame_h: int,
        lane_h: int = 22,
        margin_px: int = 1,
        sim_metric: str = "cosine",
        tau_g: float = 0.55,
        tau_w: float = 0.55,
        tau_r: float = 0.55,
        min_cov_for_label: float = 0.18,
        pair_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
        n_lines: int = 3,
        sample_frac: float | None = None,
        cov_rows: np.ndarray | None = None,
        agreement_thresh: float | None = None,
        prev_offset: int | None = None,
        search_radius: int | None = None,
    ) -> tuple[int, dict]:
        if lane_h <= 0 or frame_h <= 0:
            return 0, {"scores": {}, "best_score": None}

        if cov_rows is None:
            cov_g_row, cov_w_row, cov_r_row = BandClassifier.precompute_row_cov(
                maps,
                margin_px=margin_px,
                tau_g=tau_g,
                tau_w=tau_w,
                tau_r=tau_r,
                sample_frac=sample_frac,
            )
            cover_mat = np.stack((cov_g_row, cov_w_row, cov_r_row), axis=1)
        else:
            cover_mat = np.asarray(cov_rows, dtype=np.float32)
            if cover_mat.ndim != 2 or cover_mat.shape[1] != 3:
                raise ValueError("cov_rows must have shape (H, 3)")
        cov_g_row = cover_mat[:, 0]
        cov_w_row = cover_mat[:, 1]
        cov_r_row = cover_mat[:, 2]
        H = cover_mat.shape[0]

        w_tm, w_mb, w_tb = pair_weights
        w_sum = max(1e-6, w_tm + w_mb + w_tb)
        scores: dict[int, float] = {}
        best_offset = 0
        best_score = float("-inf")

        offset_rows = GeometryAnalyzer._get_offset_band_rows(
            H=H, lane_h=lane_h, margin_px=margin_px, n_lines=n_lines
        )
        total_offsets = list(range(len(offset_rows)))
        if not total_offsets:
            return 0, {"scores": {}, "best_score": None}

        if search_radius is None:
            search_radius = OFFSET_SEARCH_RADIUS
        search_radius = max(0, int(search_radius))

        priority_offsets: list[int] = []
        if prev_offset is not None and total_offsets:
            prev_offset = int(np.clip(prev_offset, 0, len(offset_rows) - 1))
            start_idx = max(0, prev_offset - search_radius)
            end_idx = min(len(offset_rows), prev_offset + search_radius + 1)
            priority_offsets = list(range(start_idx, end_idx))

        processed_offsets: set[int] = set()

        def evaluate_offset(offset: int) -> bool:
            nonlocal best_offset, best_score
            if offset < 0 or offset >= len(offset_rows):
                return False
            if offset in scores:
                score_val = scores[offset]
                if agreement_thresh is not None and score_val >= agreement_thresh:
                    best_offset = offset
                    best_score = score_val
                    return True
                return False

            band_rows = offset_rows[offset]
            band_scores: list[float] = []
            for row_idx_tuple in band_rows:
                if len(row_idx_tuple) < 2:
                    continue
                row_idxs = np.asarray(row_idx_tuple, dtype=int)
                row_idxs = row_idxs[(row_idxs >= 0) & (row_idxs < H)]
                if row_idxs.size < 2:
                    continue
                vecs = cover_mat[row_idxs]
                if sim_metric == "cosine":
                    if vecs.shape[0] == 3:
                        s_tm = GeometryAnalyzer._cosine_sim(vecs[0], vecs[1])
                        s_mb = GeometryAnalyzer._cosine_sim(vecs[1], vecs[2])
                        s_tb = GeometryAnalyzer._cosine_sim(vecs[0], vecs[2])
                        score = (w_tm * s_tm + w_mb * s_mb + w_tb * s_tb) / w_sum
                        band_scores.append(score)
                    else:
                        sims = [
                            GeometryAnalyzer._cosine_sim(vecs[i], vecs[i + 1])
                            for i in range(vecs.shape[0] - 1)
                        ]
                        if vecs.shape[0] > 2:
                            sims.append(GeometryAnalyzer._cosine_sim(vecs[0], vecs[-1]))
                        if sims:
                            band_scores.append(float(np.mean(sims)))
                elif sim_metric == "label":
                    labels = [
                        BandClassifier._label_from_vec(
                            vec, min_cov_for_label=min_cov_for_label
                        )
                        for vec in vecs
                    ]
                    if len(labels) == 3:
                        s_tm = 1.0 if labels[0] == labels[1] else 0.0
                        s_mb = 1.0 if labels[1] == labels[2] else 0.0
                        s_tb = 1.0 if labels[0] == labels[2] else 0.0
                        score = (w_tm * s_tm + w_mb * s_mb + w_tb * s_tb) / w_sum
                        band_scores.append(score)
                    else:
                        sims = [
                            1.0 if labels[i] == labels[i + 1] else 0.0
                            for i in range(len(labels) - 1)
                        ]
                        if len(labels) > 2:
                            sims.append(1.0 if labels[0] == labels[-1] else 0.0)
                        if sims:
                            band_scores.append(float(np.mean(sims)))
                else:
                    raise ValueError("sim_metric must be 'cosine' or 'label'")

            score_val = float(np.mean(band_scores)) if band_scores else 0.0
            scores[offset] = score_val
            if score_val > best_score:
                best_score = score_val
                best_offset = offset
            if agreement_thresh is not None and score_val >= agreement_thresh:
                return True
            return False

        def process(offset_sequence: list[int]) -> bool:
            for off in offset_sequence:
                if off in processed_offsets:
                    continue
                processed_offsets.add(off)
                if evaluate_offset(off):
                    return True
            return False

        if process(priority_offsets):
            return int(best_offset), {"scores": scores, "best_score": scores[best_offset]}

        remaining_offsets = [off for off in total_offsets if off not in processed_offsets]
        if process(remaining_offsets):
            return int(best_offset), {"scores": scores, "best_score": scores[best_offset]}

        if not scores:
            return 0, {"scores": {}, "best_score": None}

        best_offset = max(scores, key=scores.get)
        return int(best_offset), {"scores": scores, "best_score": scores[best_offset]}

class BandClassifier:
    @staticmethod
    def _pick_band_rows(y0: int, y1: int, *, margin_px: int, n_lines: int) -> list[int]:
        y_top = int(np.clip(y0 + margin_px, y0, y1 - 1))
        y_bot = int(np.clip(y1 - 1 - margin_px, y0, y1 - 1))
        if y_bot < y_top or (y_bot - y_top) < 1:
            return []
        ys = np.linspace(y_top, y_bot, num=max(2, int(n_lines)), endpoint=True)
        ys = np.unique(np.round(ys).astype(int))
        ys = ys[(ys >= y_top) & (ys <= y_bot)]
        return ys.tolist()

    @staticmethod
    def precompute_row_cov(
        maps: dict,
        *,
        margin_px: int = 1,
        tau_g: float = 0.55,
        tau_w: float = 0.55,
        tau_r: float = 0.55,
        sample_frac: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = maps["green_den"]
        b = maps["blue_den"]
        r = maps["gray_den"]
        H, W = g.shape[:2]
        x0 = margin_px
        x1 = W - margin_px if margin_px < W else W
        if sample_frac is not None:
            sample_frac = float(np.clip(sample_frac, 0.0, 1.0))
            span = max(1, int(round((x1 - x0) * sample_frac)))
            if span < (x1 - x0):
                center = (x0 + x1) // 2
                half = span // 2
                x0 = max(margin_px, center - half)
                x1 = min(W, x0 + span)
        if x1 <= x0:
            x0, x1 = 0, W
        cov_g_row = (g[:, x0:x1] >= tau_g).mean(axis=1).astype(np.float32)
        cov_w_row = (b[:, x0:x1] >= tau_w).mean(axis=1).astype(np.float32)
        cov_r_row = (r[:, x0:x1] >= tau_r).mean(axis=1).astype(np.float32)
        return cov_g_row, cov_w_row, cov_r_row

    @staticmethod
    def _label_from_vec(v: np.ndarray, *, min_cov_for_label: float = 0.18) -> LaneType:
        cov_g, cov_w, cov_r = map(float, v)
        best = max(
            [
                (cov_g, LaneType.GREEN),
                (cov_w, LaneType.WATER_PLATFORM),
                (cov_r, LaneType.ROAD),
            ],
            key=lambda t: t[0],
        )
        return best[1] if best[0] >= float(min_cov_for_label) else LaneType.UNKNOWN

    @staticmethod
    def classify_bands_from_rowcov(
        bands: list[tuple[int, int]],
        *,
        cov_g_row: np.ndarray,
        cov_w_row: np.ndarray,
        cov_r_row: np.ndarray,
        margin_px: int = 1,
        n_lines: int = 5,
        mode: str = "mean",
        min_cov_for_label: float = 0.18,
        return_debug: bool = False,
    ) -> list[LaneType] | tuple[list[LaneType], list[dict]]:
        H = cov_g_row.shape[0]
        labels: list[LaneType] = []
        debugs: list[dict] = []
        for y0, y1 in bands:
            if y1 <= y0:
                labels.append(LaneType.UNKNOWN)
                if return_debug:
                    debugs.append({"rows": [], "vec_mean": None, "label": "UNKNOWN"})
                continue
            rows = BandClassifier._pick_band_rows(
                y0, y1, margin_px=margin_px, n_lines=n_lines
            )
            if len(rows) < 2:
                labels.append(LaneType.UNKNOWN)
                if return_debug:
                    debugs.append({"rows": rows, "vec_mean": None, "label": "UNKNOWN"})
                continue
            if mode == "mean":
                vecs = np.stack(
                    [
                        np.array([cov_g_row[y], cov_w_row[y], cov_r_row[y]], np.float32)
                        for y in rows
                    ],
                    axis=0,
                )
                vec_mean = vecs.mean(axis=0)
                lab = BandClassifier._label_from_vec(
                    vec_mean, min_cov_for_label=min_cov_for_label
                )
                labels.append(lab)
                if return_debug:
                    debugs.append(
                        {
                            "rows": rows,
                            "vec_mean": vec_mean.tolist(),
                            "label": lab.name,
                        }
                    )
            elif mode == "vote":
                row_labels = [
                    BandClassifier._label_from_vec(
                        np.array(
                            [cov_g_row[y], cov_w_row[y], cov_r_row[y]], np.float32
                        ),
                        min_cov_for_label=min_cov_for_label,
                    )
                    for y in rows
                ]
                counts = {}
                for L in row_labels:
                    if L == LaneType.UNKNOWN:
                        continue
                    counts[L] = counts.get(L, 0) + 1
                lab = (
                    max(counts.items(), key=lambda kv: kv[1])[0]
                    if counts
                    else LaneType.UNKNOWN
                )
                labels.append(lab)
                if return_debug:
                    debugs.append(
                        {
                            "rows": rows,
                            "row_labels": [l.name for l in row_labels],
                            "label": lab.name,
                        }
                    )
            else:
                raise ValueError("mode must be 'mean' or 'vote'")
        return (labels, debugs) if return_debug else labels

    @staticmethod
    def shift_band_state_by_one(
        *, occ_stab, lane_kf: dict, lane_kf_stale: dict, use_kalman: bool
    ):
        if occ_stab is not None and getattr(occ_stab, "last", None) is not None:
            occ_stab.buf = np.roll(occ_stab.buf, shift=1, axis=1)
            occ_stab.buf[:, 0, :] = 0
            occ_stab.sum = np.roll(occ_stab.sum, shift=1, axis=0)
            occ_stab.sum[0, :] = 0
            occ_stab.last = np.roll(occ_stab.last, shift=1, axis=0)
            occ_stab.last[0, :] = 0
        if use_kalman and lane_kf is not None and lane_kf_stale is not None:
            kf_new = {}
            stale_new = defaultdict(int)
            for idx, kf in lane_kf.items():
                kf_new[idx + 1] = kf
            for idx, age in lane_kf_stale.items():
                stale_new[idx + 1] = age
            lane_kf.clear()
            lane_kf.update(kf_new)
            lane_kf_stale.clear()
            lane_kf_stale.update(stale_new)

@dataclass
class GameOverDetector:
    """Detect a solid orange/yellow game-over banner."""

    def __init__(
        self,
        hsv_lo=(17, 160, 200),
        hsv_hi=(30, 255, 255),
        min_frac: float = 0.0015,
        consec_needed: int = 3,
    ) -> None:
        self.lo = np.array(hsv_lo, np.uint8)
        self.hi = np.array(hsv_hi, np.uint8)
        self.min_frac = float(min_frac)
        self.consec_needed = int(consec_needed)
        self._streak = 0
        self.triggered = False
        self._k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def update(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lo, self.hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close, iterations=1)
        frac = cv2.countNonZero(mask) / mask.size
        seen = frac >= self.min_frac
        self._streak = (self._streak + 1) if seen else 0
        self.triggered = self._streak >= self.consec_needed
        return self.triggered, float(frac)

class BotStateMachine:
    def __init__(self, wait_time: float = 0.5, input_backend: Optional[InputBackend] = None):
        self.wait_time = float(wait_time)
        self._input = input_backend or DesktopInputBackend()
        self.reset()

    def reset(self):
        self.state = "start"
        self.last_change = time.perf_counter()

    def update(self, *, move_cmd: str):
        now = time.perf_counter()
        if self.state == "start":
            self._input.press('up')
            self.state = "moving"
            self.last_change = now
        elif self.state == "moving":
            if now - self.last_change >= self.wait_time:
                self.state = "static"
                self.last_change = now
        elif self.state == "static":
            if move_cmd and move_cmd != 'wait':
                self._input.press(move_cmd)
                self.state = "moving"
                self.last_change = now

class Planner:
    def __init__(self, *, occ_history_len: int = 120, cols_per_sec: float = 30.0):
        self._latest_hough_lines: list[Optional[np.ndarray]] = []
        self._tracked_occ_history: Deque[tuple[float, dict[int, np.ndarray]]] = deque(
            maxlen=max(1, int(occ_history_len))
        )
        self._latest_lane_types: list[LaneType] = []
        self._latest_character_band: Optional[int] = None
        self._latest_character_lane_type: Optional[LaneType] = None
        try:
            cps = float(cols_per_sec)
        except (TypeError, ValueError):
            cps = 30.0
        self._cols_per_sec = cps if cps > 0 else 30.0
        self._last_band_lines: list[list[dict[str, float]]] = []
        self._recent_segment_slopes: list[Deque[tuple[float, float]]] = []
        self._latest_character_rect: Optional[tuple[int, int, int, int]] = None

    def update_hough_lines(self, hough_lines: Sequence[Optional[np.ndarray]]) -> None:
        # store latest hough output for downstream planning heuristics
        self._latest_hough_lines = list(hough_lines) if hough_lines is not None else []

    def push_tracked_occ_lines(
        self,
        *,
        occ_lines: Sequence[Optional[np.ndarray]],
        tracked_indices: Sequence[int],
        t_now: float,
    ) -> None:
        snapshot: dict[int, np.ndarray] = {}
        for idx in tracked_indices:
            if not (0 <= idx < len(occ_lines)):
                continue
            line = occ_lines[idx]
            if line is None:
                continue
            snapshot[int(idx)] = np.array(line, copy=True)
        self._tracked_occ_history.append((float(t_now), snapshot))

    def update_lane_types(self, lane_types: Sequence[LaneType]) -> None:
        self._latest_lane_types = list(lane_types) if lane_types is not None else []

    def update_character_band(self, band_idx: Optional[int], lane_type: Optional[LaneType]) -> None:
        self._latest_character_band = int(band_idx) if band_idx is not None else None
        self._latest_character_lane_type = lane_type

    def plan_move(self) -> str:
        return "up"

    def project_segments(
        self,
        *,
        hough_lines: Sequence[Optional[np.ndarray]],
        segment_locations: Sequence[Optional[list[dict]]],
        hough_visual_dims: Sequence[Optional[tuple[int, int]]] | None = None,
        character_rect_x: Optional[tuple[int, int]] = None,
        future_time_s: float,
        visualize: bool = False,
    ) -> tuple[list[list[dict]], list[Optional[np.ndarray]]]:
        projections: list[list[dict]] = []
        visuals: list[Optional[np.ndarray]] = []

        self._latest_character_rect_x = character_rect_x

        hough_lines_list = list(hough_lines) if hough_lines is not None else []
        segment_locations_list = (
            list(segment_locations) if segment_locations is not None else []
        )

        hough_visual_dims_list = (
            list(hough_visual_dims) if hough_visual_dims is not None else []
        )

        extra_future_cols = max(0.0, float(self._cols_per_sec))
        now_ts = time.perf_counter()
        cutoff_ts = now_ts - 4.0

        while len(self._recent_segment_slopes) < len(hough_lines_list):
            self._recent_segment_slopes.append(deque())

        for band_idx, lines in enumerate(hough_lines_list):
            segments = (
                segment_locations_list[band_idx]
                if band_idx < len(segment_locations_list)
                else None
            )
            segments_list = list(segments) if segments else []
            projected_band: list[dict] = []
            vis: Optional[np.ndarray] = None

            col_candidates: list[float] = []
            row_candidates: list[float] = []
            projection_draws: list[tuple[float, float, float, float]] = []
            hough_line_coords: list[tuple[float, float, float, float]] = []
            slope_history = self._recent_segment_slopes[band_idx]
            while slope_history and slope_history[0][0] < cutoff_ts:
                slope_history.popleft()
            stored_slope_this_frame = False

            for seg in segments_list:
                seg_col = float(seg.get("col", 0.0))
                y0 = float(seg.get("row_start", 0.0))
                y1 = float(seg.get("row_end", y0))
                col_candidates.append(seg_col)
                row_candidates.extend([y0, y1])

            window_end_x: float | None = None
            if segments_list:
                segment_cols = [float(seg.get("col", 0.0)) for seg in segments_list]
                if segment_cols:
                    window_end_x = max(segment_cols)
            if window_end_x is None and lines is not None and len(lines):
                window_end_x = max(
                    max(float(line[0]), float(line[2]))
                    for line in lines
                    if line is not None and len(line) >= 2
                )
            if window_end_x is None:
                window_end_x = 0.0

            line_infos: list[dict[str, float]] = []

            if lines is not None and len(lines):
                for raw_line in lines:
                    if raw_line is None or len(raw_line) != 4:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in raw_line]
                    dx = x2 - x1
                    slope = (y2 - y1) / dx if abs(dx) >= 1e-6 else 0.0
                    if x2 >= x1:
                        x_right, y_right = x2, y2
                    else:
                        x_right, y_right = x1, y1
                    target_future_x = window_end_x
                    if extra_future_cols > 0:
                        target_future_x = max(target_future_x, window_end_x + extra_future_cols)
                    x_future = max(target_future_x, x_right)
                    advance_dx = x_future - x_right
                    y_future = y_right + slope * advance_dx

                    # clamp projection to window borders if a window height is known
                    window_height = None
                    if band_idx < len(hough_visual_dims_list):
                        dims = hough_visual_dims_list[band_idx]
                        if dims is not None:
                            window_height = float(dims[1])
                    if window_height is None and row_candidates:
                        window_height = max(row_candidates)
                    if window_height is None:
                        window_height = 0.0
                    if window_height > 0 and abs(dx) >= 1e-6:
                        if y_future < 0:
                            needed_dy = -y_right
                            clipped_dx = needed_dy / (slope) if abs(slope) > 1e-6 else None
                            if clipped_dx is not None and clipped_dx >= 0:
                                clipped_dx = min(advance_dx, clipped_dx)
                                x_future = x_right + clipped_dx
                                y_future = y_right + slope * clipped_dx
                        elif y_future > window_height:
                            needed_dy = window_height - y_right
                            clipped_dx = needed_dy / (slope) if abs(slope) > 1e-6 else None
                            if clipped_dx is not None and clipped_dx >= 0:
                                clipped_dx = min(advance_dx, clipped_dx)
                                x_future = x_right + clipped_dx
                                y_future = y_right + slope * clipped_dx

                    col_candidates.extend([x1, x2])
                    row_candidates.extend([y1, y2])
                    hough_line_coords.append((x1, y1, x2, y2))
                    line_infos.append(
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "dx": dx,
                            "slope": slope,
                            "x_right": x_right,
                            "y_right": y_right,
                        }
                    )

            def _line_y_at(info: dict[str, float], x_val: float) -> float:
                dx_val = info["x2"] - info["x1"]
                if abs(dx_val) < 1e-6:
                    return float(info["y2"])
                t = (x_val - info["x1"]) / dx_val
                return info["y1"] + t * (info["y2"] - info["y1"])

            window_height: Optional[float] = None
            if band_idx < len(hough_visual_dims_list):
                dims = hough_visual_dims_list[band_idx]
                if dims is not None:
                    window_height = float(dims[1])
            if window_height is None and row_candidates:
                window_height = float(max(row_candidates))

            used_lines: set[int] = set()
            segment_to_line: dict[int, Optional[int]] = {}

            for seg_idx, seg in enumerate(segments_list):
                seg_col = float(seg.get("col", window_end_x))
                y0 = float(seg.get("row_start", 0.0))
                y1 = float(seg.get("row_end", y0))
                seg_center = 0.5 * (y0 + y1)
                best_idx = None
                best_dist = float("inf")
                for idx, info in enumerate(line_infos):
                    y_pred = _line_y_at(info, seg_col)
                    dist = abs(y_pred - seg_center)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                segment_to_line[seg_idx] = best_idx
                if best_idx is not None:
                    used_lines.add(best_idx)

            def _clamp_rows(start_val: float, end_val: float) -> tuple[float, float]:
                if window_height is None or window_height <= 0:
                    return start_val, end_val
                start_clamped = max(0.0, min(window_height, start_val))
                end_clamped = max(0.0, min(window_height, end_val))
                return start_clamped, end_clamped

            def _extend_with_slope(
                start_row: float,
                end_row: float,
                origin_col: float,
                slope_val: float,
                target_x: float,
            ) -> tuple[float, float, float]:
                dx_future = target_x - origin_col
                if slope_val != 0.0 and window_height is not None and window_height > 0.0:
                    y_low = min(start_row, end_row)
                    y_high = max(start_row, end_row)
                    delta_target = slope_val * dx_future
                    delta_min = -y_low
                    delta_max = window_height - y_high
                    delta_clamped = max(delta_min, min(delta_max, delta_target))
                    dx_future = delta_clamped / slope_val
                delta = slope_val * dx_future
                new_start = start_row + delta
                new_end = end_row + delta
                new_start, new_end = _clamp_rows(new_start, new_end)
                return new_start, new_end, origin_col + dx_future

            # Extend matched segments first
            for seg_idx, seg in enumerate(segments_list):
                seg_col = float(seg.get("col", window_end_x))
                y0 = float(seg.get("row_start", 0.0))
                y1 = float(seg.get("row_end", y0))
                height = y1 - y0
                assigned_line_idx = segment_to_line.get(seg_idx)
                slope_val = 0.0
                derived_from_line = False
                if assigned_line_idx is not None:
                    slope_val = float(line_infos[assigned_line_idx]["slope"])
                    if abs(slope_val) >= 1e-6:
                        derived_from_line = True
                if abs(slope_val) < 1e-6 and slope_history:
                    slope_val = slope_history[-1][1]
                target_future_x = max(seg_col + extra_future_cols, window_end_x)
                new_start, new_end, col_future = _extend_with_slope(
                    y0,
                    y1,
                    seg_col,
                    slope_val,
                    target_future_x,
                )
                if int(round(min(new_start, new_end))) != int(round(max(new_start, new_end))):
                    projected_band.append(
                        {
                            "row_start": int(round(min(new_start, new_end))),
                            "row_end": int(round(max(new_start, new_end))),
                            "col": int(round(col_future)),
                        }
                    )
                    col_candidates.append(col_future)
                    row_candidates.extend([new_start, new_end])
                    center_now = y0 + 0.5 * height
                    center_future = center_now + slope_val * (col_future - seg_col)
                    projection_draws.append(
                        (seg_col, center_now, col_future, center_future)
                    )
                    if (
                        derived_from_line
                        and not stored_slope_this_frame
                        and abs(slope_val) >= 1e-6
                    ):
                        slope_history.append((now_ts, slope_val))
                        stored_slope_this_frame = True

            # Extend leftover lines not matched to segments
            unused_lines = [
                (idx, info) for idx, info in enumerate(line_infos) if idx not in used_lines
            ]
            for idx, info in unused_lines:
                x_right = info["x_right"]
                y_right = info["y_right"]
                slope = info["slope"]
                target_future_x = max(window_end_x, x_right + extra_future_cols)
                advance_dx = target_future_x - x_right
                y_future = y_right + slope * advance_dx
                if window_height is not None and window_height > 0 and abs(info["dx"]) >= 1e-6:
                    if y_future < 0:
                        needed_dy = -y_right
                        clipped_dx = needed_dy / slope if abs(slope) > 1e-6 else None
                        if clipped_dx is not None and clipped_dx >= 0:
                            clipped_dx = min(advance_dx, clipped_dx)
                            target_future_x = x_right + clipped_dx
                            y_future = y_right + slope * clipped_dx
                    elif y_future > window_height:
                        needed_dy = window_height - y_right
                        clipped_dx = needed_dy / slope if abs(slope) > 1e-6 else None
                        if clipped_dx is not None and clipped_dx >= 0:
                            clipped_dx = min(advance_dx, clipped_dx)
                            target_future_x = x_right + clipped_dx
                            y_future = y_right + slope * clipped_dx

                row_start = min(y_right, y_future)
                row_end = max(y_right, y_future)
                col_future = target_future_x

                projected_band.append(
                    {
                        "row_start": int(round(row_start)),
                        "row_end": int(round(row_end)),
                        "col": int(round(col_future)),
                    }
                )

                col_candidates.extend([x_right, col_future])
                row_candidates.extend([y_right, y_future])
                projection_draws.append(
                    (x_right, y_right, col_future, y_future)
                )

            if visualize and (hough_line_coords or projection_draws):
                width = height = None
                if band_idx < len(hough_visual_dims_list):
                    dims = hough_visual_dims_list[band_idx]
                    if dims is not None:
                        base_width = max(1, int(round(dims[0])))
                        width = max(1, int(round(base_width + max(0.0, extra_future_cols))))
                        height = int(dims[1])
                if width is None or height is None:
                    max_col = max(col_candidates) if col_candidates else window_end_x
                    max_col = max_col if max_col is not None else 0.0
                    max_row = max(row_candidates) if row_candidates else 0.0
                    width = max(1, int(math.ceil(max(0.0, max_col)) + extra_future_cols) + 2)
                    height = max(1, int(math.ceil(max(0.0, max_row))) + 1)
                max_proj_col = max(col_candidates) if col_candidates else 0.0
                width = max(width or 1, int(math.ceil(max(0.0, max_proj_col))) + 2)
                lines_for_visual: list[tuple[int, int, int, int]] = [
                    (
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    )
                    for x1, y1, x2, y2 in hough_line_coords
                ]
                vis = render_lines_visual_base(lines_for_visual, width, height)
                if vis is None and width > 0 and height > 0:
                    vis = np.zeros((height, width, 3), dtype=np.uint8)
                if vis is not None:
                    for x_start, y_start, x_end, y_end in projection_draws:
                        xs = int(round(x_start))
                        ys = int(round(y_start))
                        xe = int(round(x_end))
                        ye = int(round(y_end))
                        xs = max(0, min(width - 1, xs))
                        ys = max(0, min(height - 1, ys))
                        xe = max(0, min(width - 1, xe))
                        ye = max(0, min(height - 1, ye))
                        cv2.line(vis, (xs, ys), (xe, ye), (0, 255, 0), 1, cv2.LINE_AA)
                    if segments:
                        for seg in segments:
                            if seg is None:
                                continue
                            seg_col = int(round(float(seg.get("col", 0.0))))
                            y0 = float(seg.get("row_start", 0.0))
                            y1 = float(seg.get("row_end", y0))
                            seg_row = int(round((y0 + y1) * 0.5))
                            seg_col = max(0, min(width - 1, seg_col))
                            seg_row = max(0, min(height - 1, seg_row))
                            cv2.circle(vis, (seg_col, seg_row), 3, (0, 0, 255), 1, cv2.LINE_AA)

            projections.append(projected_band)
            visuals.append(vis)

        return projections, visuals



class VelocityEstimator:
    def __init__(
        self,
        window_s: float = 2.0,
        *,
        time_on_x: bool = True,
        max_horizontal_run_px: int | None = None,
        repeat_suppression: int = 4,
        hough_rho: float = 1.0,
        hough_theta: float = np.pi / 180.0,
        hough_threshold: int = 25,
        hough_min_line_length: float = 10.0,
        hough_max_line_gap: float = 4.0,
        future_projection_s: float = 2.0,
        sobel_temporal_and_frames: int = 1,
        sampling_rate_hz: float = 30.0,
        resample_method: str = "nearest",
    ):
        self.window_s = float(window_s)
        self._time_axis = 'x' if time_on_x else 'y'
        max_run = None
        if max_horizontal_run_px is not None:
            try:
                max_run_val = int(max_horizontal_run_px)
            except (TypeError, ValueError):
                max_run_val = 0
            if max_run_val > 0:
                max_run = max_run_val
        self._max_run_px = max_run
        self._repeat_suppression = max(0, int(repeat_suppression))
        try:
            rho_val = float(hough_rho)
        except (TypeError, ValueError):
            rho_val = 1.0
        self._hough_rho = rho_val if rho_val > 0 else 1.0
        try:
            theta_val = float(hough_theta)
        except (TypeError, ValueError):
            theta_val = float(np.pi / 180.0)
        self._hough_theta = theta_val if theta_val > 0 else float(np.pi / 180.0)
        try:
            threshold_val = int(hough_threshold)
        except (TypeError, ValueError):
            threshold_val = 1
        self._hough_threshold = max(1, threshold_val)
        try:
            min_len_val = float(hough_min_line_length)
        except (TypeError, ValueError):
            min_len_val = 1.0
        self._hough_min_line_length = max(1.0, min_len_val)
        try:
            max_gap_val = float(hough_max_line_gap)
        except (TypeError, ValueError):
            max_gap_val = 0.0
        self._hough_max_line_gap = max(0.0, max_gap_val)
        try:
            sobel_and_val = int(sobel_temporal_and_frames)
        except (TypeError, ValueError):
            sobel_and_val = 1
        self._sobel_temporal_and_frames = max(1, sobel_and_val)
        try:
            sample_rate = float(sampling_rate_hz)
        except (TypeError, ValueError):
            sample_rate = 25.0
        if sample_rate <= 0.0 or not math.isfinite(sample_rate):
            sample_rate = 25.0
        self._sampling_rate_hz = sample_rate
        method = str(resample_method or "nearest").strip().lower()
        self._resample_method = method if method in ("nearest", "linear") else "nearest"
        self._rows: list[Deque[np.ndarray]] = []
        self._times: list[Deque[float]] = []
        self._widths: list[int] = []
        self._raw_history: list[Deque[np.ndarray]] = []
        self._latest_hough_lines: list[Optional[np.ndarray]] = []
        self._latest_sobel_visual: list[Optional[np.ndarray]] = []
        self._latest_actual_hough_visual: list[Optional[np.ndarray]] = []
        self._latest_hough_line_details: list[Optional[Dict[str, Any]]] = []
        self._hough_line_history: list[Deque[Dict[str, Any]]] = []
        self._persistent_line_sets: list[dict[str, Optional[np.ndarray]]] = []

    def push_occ_lines(self, occ_lines: List[np.ndarray], t_now: float) -> None:
        self._ensure_bands(len(occ_lines))
        t_now = float(t_now)
        for band, line in enumerate(occ_lines):
            if line is None:
                continue
            li_raw = self._coerce_binary_1d(line)
            if li_raw.size == 0:
                continue
            width = int(li_raw.shape[0])
            if self._widths[band] != width:
                self._rows[band].clear()
                self._times[band].clear()
                self._raw_history[band].clear()
                self._latest_hough_lines[band] = None
                self._latest_sobel_visual[band] = None
                self._latest_actual_hough_visual[band] = None
                self._latest_hough_line_details[band] = None
                self._hough_line_history[band].clear()
                self._persistent_line_sets[band] = {"pos": None, "neg": None}
                self._widths[band] = width

            line_img = (li_raw > 0).astype(np.uint8) * 255
            self._rows[band].append(line_img)
            self._times[band].append(t_now)
            self._latest_hough_lines[band] = None
            self._latest_hough_line_details[band] = None
            self._hough_line_history[band].clear()
            self._latest_actual_hough_visual[band] = None
            self._trim_band(band)

    def shift_bands_down_by_one(self) -> None:
        """Shift per-band buffers down by one slot, clearing the newest slot."""
        if not self._rows:
            return

        def _shift(lst, new_item):
            lst.insert(0, new_item)
            lst.pop()

        _shift(self._rows, deque())
        _shift(self._times, deque())
        _shift(self._widths, 0)
        _shift(self._raw_history, deque())
        _shift(self._latest_hough_lines, None)
        _shift(self._latest_sobel_visual, None)
        _shift(self._latest_actual_hough_visual, None)
        _shift(self._latest_hough_line_details, None)
        _shift(self._hough_line_history, deque())
        _shift(self._persistent_line_sets, {"pos": None, "neg": None})

        if hasattr(self, "_latest_line_map"):
            old_map = getattr(self, "_latest_line_map", {})
            new_map = {}
            band_count = len(self._rows)
            for idx in range(band_count):
                if idx == 0:
                    new_map[idx] = None
                else:
                    new_map[idx] = old_map.get(idx - 1)
            self._latest_line_map = new_map

    def visualize_image(
        self,
        band: int,
        *,
        orientation: str | None = None,
        x_scale: int = 1,
        y_scale: int = 1,
    ) -> np.ndarray:
        image = self._get_image(band)
        if image is None:
            return np.zeros((40, 200, 3), np.uint8)
        orient = (orientation or self._time_axis).lower()
        if orient not in ('x', 'y'):
            orient = self._time_axis
        xt = image.T if orient == 'x' else image
        if xt.dtype != np.uint8:
            xt = np.clip(np.rint(xt), 0, 255).astype(np.uint8)
        vis = cv2.cvtColor(xt, cv2.COLOR_GRAY2BGR)
        if x_scale != 1 or y_scale != 1:
            vis = ImageOps.resize(
                vis,
                (
                    vis.shape[1] * max(1, int(x_scale)),
                    vis.shape[0] * max(1, int(y_scale)),
                ),
                interpolation=cv2.INTER_NEAREST,
            )
        return vis

    def probabilistic_hough( 
        self,
        band: int,
        *,
        orientation: str | None = None,
        x_scale: int = 1,
        y_scale: int = 1,
        return_visual: bool = True,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[list[tuple[int, int]]]]:

        if not self._valid_band(band):
            return None, None, None

        self._latest_sobel_visual[band] = None

        orient = (orientation or self._time_axis).lower()
        if orient not in ("x", "y"):
            orient = self._time_axis

        image = self._get_image_uniform_time(
            band,
            self._sampling_rate_hz,
            orientation=orient,
            method=self._resample_method,
        )
        if image is None:
            self._latest_hough_lines[band] = None
            empty_vis = np.zeros((40, 200, 3), np.uint8) if return_visual else None
            self._latest_sobel_visual[band] = None
            self._latest_hough_line_details[band] = None
            return None, empty_vis, None

        xt = image
        if xt.dtype != np.uint8:
            xt_u8 = np.clip(np.rint(xt), 0, 255).astype(np.uint8)
        else:
            xt_u8 = xt.copy()

        if xt_u8.ndim == 1:
            xt_u8 = xt_u8.reshape(1, -1)

        if xt_u8.size == 0:
            self._latest_hough_lines[band] = None
            empty_vis = np.zeros((40, 200, 3), np.uint8) if return_visual else None
            self._latest_sobel_visual[band] = None
            self._latest_hough_line_details[band] = None
            return None, empty_vis, None

        if int(np.max(xt_u8)) > 0:
            _, edge_map = cv2.threshold(xt_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            edge_map = xt_u8.copy()

        # Sobel (vertical) -> abs -> normalize
        sobel_v = cv2.Sobel(edge_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        sobel_abs = np.abs(sobel_v)
        sobel_norm = cv2.normalize(sobel_abs, None, 0, 255, cv2.NORM_MINMAX)
        if sobel_norm.dtype != np.uint8:
            sobel_norm = np.clip(np.rint(sobel_norm), 0, 255).astype(np.uint8)

        # Sobel -> threshold for Hough input tune0
        edge_map = sobel_norm.copy()
        _, edge_map = cv2.threshold(edge_map, 100, 255, cv2.THRESH_BINARY)
        max_run = getattr(self, "_max_run_px", None)
        if max_run is not None and max_run > 0:
            horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max_run, 1))
            long_runs = cv2.morphologyEx(edge_map, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
            edge_map = cv2.subtract(edge_map, long_runs)

        # # Morphological closing (dilate then erode) to bridge tiny gaps
        # morph_kernel = np.ones((3, 3), np.uint8)
        # edge_map = cv2.dilate(edge_map, morph_kernel, iterations=1)#tune0

        base_h, base_w = edge_map.shape[:2]

        # Optionally downscale edges before Hough for performance (kept disabled)
        hough_input = edge_map
        scale_back_x = 1.0
        scale_back_y = 1.0

        # Hough (probabilistic) ---------------
        lines = cv2.HoughLinesP(
            hough_input,
            rho=self._hough_rho,
            theta=self._hough_theta,
            threshold=int(self._hough_threshold),
            minLineLength=float(self._hough_min_line_length),
            maxLineGap=float(self._hough_max_line_gap),
        )
        if lines is None:
            lines_arr = None
        else:
            lines_arr = lines.reshape(-1, 4).astype(np.float32)
            if scale_back_x != 1.0 or scale_back_y != 1.0:
                max_x = float(max(0, base_w - 1))
                max_y = float(max(0, base_h - 1))
                lines_arr[:, [0, 2]] = np.clip(
                    lines_arr[:, [0, 2]] * scale_back_x, 0.0, max_x
                )
                lines_arr[:, [1, 3]] = np.clip(
                    lines_arr[:, [1, 3]] * scale_back_y, 0.0, max_y
                )
            lines_arr = np.rint(lines_arr).astype(int)

        # Pre-filter: reject near-horizontal, zero-length
        if lines_arr is not None and len(lines_arr) > 0:
            kept = []
            for x1, y1, x2, y2 in lines_arr:
                dy = y2 - y1
                dx = x2 - x1
                if dx == 0 and dy == 0:
                    continue
                length = math.hypot(float(dx), float(dy))
                if length <= 0:
                    continue
                slope_y = abs(float(dy)) / length
                if slope_y < 0.2:  # near-horizontal -> skip
                    continue
                kept.append([x1, y1, x2, y2])
            lines_arr = np.array(kept, dtype=int) if kept else None

        # Work within current Sobel extent
        target_w = base_w
        target_h = base_h

        sobel_canvas = hough_input.copy()
        sobel_vis = cv2.cvtColor(sobel_canvas, cv2.COLOR_GRAY2BGR)
        if x_scale != 1 or y_scale != 1:
            sobel_vis = ImageOps.resize(
                sobel_vis,
                (
                    sobel_vis.shape[1] * max(1, int(x_scale)),
                    sobel_vis.shape[0] * max(1, int(y_scale)),
                ),
                interpolation=cv2.INTER_NEAREST,
            )
        segments: list[tuple[int, int]] = []
        latest_column = hough_input[:, -1].copy() if hough_input.shape[1] > 0 else None
        if latest_column is not None and latest_column.size > 0:
            mask = latest_column > 0
            if np.any(mask):
                mask_int = mask.astype(np.int8)
                transitions = np.diff(mask_int)
                starts = np.where(transitions == 1)[0] + 1
                ends = np.where(transitions == -1)[0]
                if mask[0]:
                    starts = np.concatenate(([0], starts))
                if mask[-1]:
                    ends = np.concatenate((ends, [mask.size - 1]))
                segments = list(zip(starts, ends))
                if segments:
                    scale_x = max(1, int(x_scale))
                    scale_y = max(1, int(y_scale))
                    x_recent = max(0, sobel_vis.shape[1] - scale_x)
                    for seg_start, seg_end in segments:
                        cy = (seg_start + seg_end) * 0.5
                        sy = int(
                            np.clip(round(cy * scale_y), 0, sobel_vis.shape[0] - 1)
                        )
                        cv2.circle(sobel_vis, (x_recent, sy), 2, (0, 0, 255), 1)
        self._latest_sobel_visual[band] = sobel_vis

        # Prepare Sobel map for overlap checks
        edge_padded = edge_map.copy()

        # --- Use line_map + AND-overlap to decide which lines to keep ---
        line_map = None
        kept_indices: list[int] = []

        if lines_arr is not None and len(lines_arr):
            # Configurable thresholds ---------------------
            min_px_abs = int(getattr(self, "_line_overlap_min_px", 20))          # minimal overlapped pixels with Sobel
            min_frac = float(getattr(self, "_line_overlap_min_frac", 0.25))      # minimal fraction of line length 

            for idx, (x1, y1, x2, y2) in enumerate(lines_arr):
                # Rasterize this candidate line
                tmp = np.zeros((target_h, target_w), dtype=np.uint8)
                cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1, cv2.LINE_8)

                # Overlap this line with the Sobel edges
                overlap = cv2.bitwise_and(tmp, edge_padded)

                # Count overlapped pixels
                overlap_count = int(np.count_nonzero(overlap))

                # Relative requirement by geometric length (optional)
                length_px = max(1.0, math.hypot(float(x2 - x1), float(y2 - y1)))
                ok_rel = (overlap_count >= (min_frac * length_px)) if min_frac > 0.0 else True
                ok_abs = (overlap_count >= min_px_abs)

                if ok_abs and ok_rel:
                    kept_indices.append(idx)

        lines_filtered = None
        lines_filtered_raw = None
        if lines_arr is not None and kept_indices:
            kept_raw = lines_arr[np.array(kept_indices, dtype=int)]
            lines_filtered_raw = kept_raw.copy()
            lines_filtered = self._extend_lines_to_canvas(
                kept_raw,
                target_w,
                target_h,
            )
        lines_final = lines_filtered.copy() if lines_filtered is not None else None

        # --- Persistent line set management (for Visualization) ---
        def _line_slope_sign(line: Sequence[int], eps: float = 1e-6) -> Optional[str]:
            x1, y1, x2, y2 = [float(v) for v in line]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < eps:
                if abs(dy) < eps:
                    return None
                return "pos" if dy > 0 else "neg"
            slope = dy / dx
            if slope > eps:
                return "pos"
            if slope < -eps:
                return "neg"
            return None

        stored_sets = self._persistent_line_sets[band]
        if not isinstance(stored_sets, dict):
            stored_sets = {"pos": None, "neg": None}
            self._persistent_line_sets[band] = stored_sets

        line_sets_current = {
            "pos": stored_sets.get("pos").copy() if stored_sets.get("pos") is not None else None,
            "neg": stored_sets.get("neg").copy() if stored_sets.get("neg") is not None else None,
        }
        line_sets_updated = {
            key: arr.copy() if arr is not None else None for key, arr in line_sets_current.items()
        }

        existing_metrics = {"pos": None, "neg": None}
        for sign in ("pos", "neg"):
            arr = line_sets_current[sign]
            if arr is None or len(arr) == 0:
                continue
            existing_line = self._select_rightmost_line(arr, target_w, target_h)
            if existing_line is not None:
                existing_metrics[sign] = self._line_right_order_value(existing_line)

        lines_by_sign: dict[str, list[np.ndarray]] = {"pos": [], "neg": []}
        if lines_final is not None and len(lines_final):
            for line in lines_final:
                sign = _line_slope_sign(line)
                if sign in lines_by_sign:
                    lines_by_sign[sign].append(np.asarray(line, dtype=int))

        replace_tol = float(getattr(self, "_line_replace_metric_tol", 8.0))

        for sign in ("pos", "neg"):
            candidates = lines_by_sign[sign]
            if not candidates:
                continue
            arr_updated = line_sets_updated[sign]
            metrics_list: list[Optional[float]] = []
            if arr_updated is not None and len(arr_updated):
                for stored_line in arr_updated:
                    metrics_list.append(self._line_right_order_value(stored_line))
            threshold = (
                max((val for val in metrics_list if val is not None), default=None)
                if metrics_list
                else None
            )
            for line in candidates:
                order_val = self._line_right_order_value(line)
                if order_val is None:
                    continue
                replaced = False
                if arr_updated is not None and metrics_list:
                    for idx, val in enumerate(metrics_list):
                        if val is None:
                            continue
                        delta = order_val - val
                        if -replace_tol <= delta <= replace_tol:
                            arr_updated[idx] = line
                            metrics_list[idx] = order_val
                            replaced = True
                            break
                if replaced:
                    continue
                # threshold may need refresh after replacements
                threshold = (
                    max((val for val in metrics_list if val is not None), default=None)
                    if metrics_list
                    else None
                )
                should_add = False
                if threshold is None or arr_updated is None or len(arr_updated) == 0:
                    should_add = True
                else:
                    if order_val > threshold + 1e-6:
                        should_add = True
                if should_add:
                    if arr_updated is None:
                        arr_updated = np.asarray([line], dtype=int)
                        metrics_list = [order_val]
                    else:
                        arr_updated = np.vstack([arr_updated, line])
                        metrics_list.append(order_val)
                    threshold = max((val for val in metrics_list if val is not None), default=None)
            line_sets_updated[sign] = arr_updated

        line_sets_for_display = {
            sign: (line_sets_updated[sign].copy() if line_sets_updated[sign] is not None else None)
            for sign in ("pos", "neg")
        }
        rightmost_line = self._select_rightmost_line(lines_final, target_w, target_h)

        lines_for_output: list[list[int]] = []
        if lines_final is not None and len(lines_final):
            lines_for_output.extend(
                [list(map(int, line_vals)) for line_vals in lines_final]
            )
        for arr in line_sets_for_display.values():
            if arr is None:
                continue
            lines_for_output.extend([list(map(int, vals)) for vals in arr])

        base_canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if x_scale != 1 or y_scale != 1:
            base_canvas = ImageOps.resize(
                base_canvas,
                (
                    base_canvas.shape[1] * max(1, int(x_scale)),
                    base_canvas.shape[0] * max(1, int(y_scale)),
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        actual_vis = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        lines_white = self._render_lines_visual(lines_filtered, target_w, target_h, x_scale, y_scale)
        if lines_white is not None:
            actual_vis = lines_white
        for arr in line_sets_for_display.values():
            if arr is None:
                continue
            for persistent_line in arr:
                self._highlight_line_on_visual(
                    actual_vis,
                    persistent_line,
                    x_scale,
                    y_scale,
                    color=(0, 255, 255),
                    thickness=1,
                )
        self._latest_actual_hough_visual[band] = actual_vis

        # Persist line_map for programmatic use (not visualized)
        if not hasattr(self, "_latest_line_map"):
            self._latest_line_map = {}
        if lines_final is not None and len(lines_final):
            line_map = -np.ones((target_h, target_w), dtype=np.int32)
            for idx, (x1, y1, x2, y2) in enumerate(lines_final):
                tmp = np.zeros((target_h, target_w), dtype=np.uint8)
                cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1, cv2.LINE_8)
                assign_mask = (tmp > 0) & (line_map < 0)
                if np.any(assign_mask):
                    line_map[assign_mask] = idx
            self._latest_line_map[band] = line_map
        else:
            self._latest_line_map[band] = None

        vis = None
        if return_visual:
            vis = actual_vis.copy()


        # Shift persistent line sets left by 1px for next frame
        shifted_sets: dict[str, Optional[np.ndarray]] = {"pos": None, "neg": None}
        for sign in ("pos", "neg"):
            arr = line_sets_updated[sign]
            if arr is None or len(arr) == 0:
                continue
            shifted = arr.astype(float)
            shifted[:, [0, 2]] -= 1.0
            extended_shifted: list[list[int]] = []
            for line_vals in shifted:
                ext = self._extend_line_to_canvas(
                    line_vals,
                    target_w,
                    target_h,
                )
                if ext is not None:
                    extended_shifted.append(ext)
            if extended_shifted:
                shifted_sets[sign] = np.asarray(extended_shifted, dtype=int)
            else:
                shifted_sets[sign] = None
        self._persistent_line_sets[band] = shifted_sets

        lines_out = (
            np.asarray(lines_for_output, dtype=int) if lines_for_output else None
        )

        # Update latest kept lines
        if lines_out is None or len(lines_out) == 0:
            self._latest_hough_lines[band] = None
        else:
            self._latest_hough_lines[band] = lines_out.copy()
        col_idx = int(hough_input.shape[1] - 1) if hough_input.shape[1] > 0 else 0
        segments_info = (
            [
                {"row_start": int(s), "row_end": int(e), "col": col_idx}
                for s, e in segments
            ]
            if segments
            else None
        )
        return lines_out, vis, segments_info


    def _extend_lines_to_canvas(
        self,
        lines: np.ndarray,
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        if lines is None or len(lines) == 0:
            return None
        width = max(1, int(width))
        height = max(1, int(height))
        extended: list[list[int]] = []
        for line in lines:
            ext_line = self._extend_line_to_canvas(line, width, height)
            if ext_line is not None:
                extended.append(ext_line)
        if not extended:
            return None
        return np.array(extended, dtype=int)

    def _render_lines_visual(
        self,
        lines: Optional[np.ndarray],
        width: int,
        height: int,
        x_scale: int = 1,
        y_scale: int = 1,
    ) -> Optional[np.ndarray]:
        return render_lines_visual_base(lines, width, height, x_scale, y_scale)

    @staticmethod
    def _line_top_intersection_x(line: Sequence[int]) -> Optional[tuple[float, int]]:
        if line is None or len(line) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        eps = 1e-6
        target_y = 0.0
        if abs(dy) < eps:
            if abs(y1 - target_y) < eps:
                return max(x1, x2), 0
            return None
        t = (target_y - y1) / dy
        x = x1 + t * (x2 - x1)
        slope_sign = 0
        if abs(dx) >= eps:
            slope = dy / dx
            if slope > 0:
                slope_sign = 1
            elif slope < 0:
                slope_sign = -1
            else:
                slope_sign = 0
        else:
            slope_sign = 1 if dy > 0 else -1
        return x, slope_sign

    @classmethod
    def _line_right_order_value(cls, line: Sequence[int]) -> Optional[float]:
        info = cls._line_top_intersection_x(line)
        if info is None:
            return None
        x_val, slope_sign = info
        return x_val

    @classmethod
    def _select_rightmost_line(
        cls,
        lines: Optional[np.ndarray],
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        if lines is None or len(lines) == 0 or width <= 0 or height <= 0:
            return None
        best_line = None
        best_metric = -math.inf
        for line in lines:
            metric = cls._line_right_order_value(line)
            if metric is None:
                continue
            if metric > best_metric:
                best_metric = metric
                best_line = np.asarray(line, dtype=float)
        return best_line

    @staticmethod
    def _highlight_line_on_visual(
        vis: np.ndarray,
        line: Sequence[int] | np.ndarray,
        x_scale: int,
        y_scale: int,
        color: tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2,
    ) -> None:
        if vis is None or line is None:
            return
        sx = max(1, int(x_scale))
        sy = max(1, int(y_scale))
        x1, y1, x2, y2 = [int(round(v)) for v in line]
        p1 = (int(round(x1 * sx)), int(round(y1 * sy)))
        p2 = (int(round(x2 * sx)), int(round(y2 * sy)))
        cv2.line(vis, p1, p2, color, thickness, cv2.LINE_8)

    @staticmethod
    def _extend_line_to_canvas(
        line: Sequence[int],
        width: int,
        height: int,
    ) -> Optional[list[int]]:
        if width <= 0 or height <= 0:
            return None
        x1, y1, x2, y2 = [float(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        eps = 1e-6
        x_min = 0.0
        y_min = 0.0
        x_max = float(max(0, width - 1))
        y_max = float(max(0, height - 1))
        if abs(dx) < eps and abs(dy) < eps:
            x = float(np.clip(x1, x_min, x_max))
            y = float(np.clip(y1, y_min, y_max))
            val = int(round(x)), int(round(y))
            return [val[0], val[1], val[0], val[1]]
        points: list[tuple[float, float]] = []
        if abs(dx) > eps:
            for x_edge in (x_min, x_max):
                t = (x_edge - x1) / dx
                y_edge = y1 + t * dy
                if y_min - eps <= y_edge <= y_max + eps:
                    points.append((x_edge, float(np.clip(y_edge, y_min, y_max))))
        if abs(dy) > eps:
            for y_edge in (y_min, y_max):
                t = (y_edge - y1) / dy
                x_edge = x1 + t * dx
                if x_min - eps <= x_edge <= x_max + eps:
                    points.append((float(np.clip(x_edge, x_min, x_max)), y_edge))
        if not points:
            x1c = float(np.clip(x1, x_min, x_max))
            y1c = float(np.clip(y1, y_min, y_max))
            x2c = float(np.clip(x2, x_min, x_max))
            y2c = float(np.clip(y2, y_min, y_max))
            return [int(round(x1c)), int(round(y1c)), int(round(x2c)), int(round(y2c))]
        unique: list[tuple[float, float]] = []
        for x, y in points:
            if not any(abs(x - ux) < 1e-4 and abs(y - uy) < 1e-4 for ux, uy in unique):
                unique.append((x, y))
        if len(unique) == 1:
            unique.append(unique[0])
        if len(unique) > 2:
            best_pair: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
            best_dist = -1.0
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    xi, yi = unique[i]
                    xj, yj = unique[j]
                    dist = (xi - xj) ** 2 + (yi - yj) ** 2
                    if dist > best_dist:
                        best_dist = dist
                        best_pair = ((xi, yi), (xj, yj))
            if best_pair is None:
                p1, p2 = unique[0], unique[-1]
            else:
                p1, p2 = best_pair
        else:
            p1, p2 = unique[0], unique[-1]

        def clamp_point(pt: tuple[float, float]) -> tuple[int, int]:
            px = float(np.clip(pt[0], x_min, x_max))
            py = float(np.clip(pt[1], y_min, y_max))
            return int(round(px)), int(round(py))

        a = clamp_point(p1)
        b = clamp_point(p2)
        return [a[0], a[1], b[0], b[1]]


    def get_hough_line_history(self, band: int, window_s: float = 10.0) -> list[Dict[str, Any]]:
        if not self._valid_band(band):
            return []
        history = self._hough_line_history[band]
        if not history:
            return []
        now = float(time.time())
        window = max(0.0, float(window_s))
        cutoff = now - window
        result: list[Dict[str, Any]] = []
        for packet in history:
            ts = float(packet.get("timestamp", 0.0))
            if ts >= cutoff:
                lines = packet.get("lines", [])
                result.append(
                    {
                        "band": int(packet.get("band", band)),
                        "orientation": packet.get("orientation"),
                        "timestamp": ts,
                        "line_count": int(packet.get("line_count", len(lines))),
                        "lines": [dict(line) for line in lines],
                    }
                )
        return result

    def get_latest_sobel_visual(self, band: int) -> Optional[np.ndarray]:
        if not self._valid_band(band):
            return None
        sobel_vis = self._latest_sobel_visual[band]
        if sobel_vis is None:
            return None
        return sobel_vis.copy()

    def get_latest_actual_hough_visual(self, band: int) -> Optional[np.ndarray]:
        if not self._valid_band(band):
            return None
        vis = self._latest_actual_hough_visual[band]
        if vis is None:
            return None
        return vis.copy()

    def _ensure_bands(self, n: int) -> None:
        grow = n - len(self._rows)
        if grow <= 0:
            return
        self._rows += [deque() for _ in range(grow)]
        self._times += [deque() for _ in range(grow)]
        self._widths += [0 for _ in range(grow)]
        self._raw_history += [deque() for _ in range(grow)]
        self._latest_hough_lines += [None for _ in range(grow)]
        self._latest_sobel_visual += [None for _ in range(grow)]
        self._latest_actual_hough_visual += [None for _ in range(grow)]
        self._latest_hough_line_details += [None for _ in range(grow)]
        self._hough_line_history += [deque() for _ in range(grow)]
        self._persistent_line_sets += [{"pos": None, "neg": None} for _ in range(grow)]

    def _get_image(self, band: int) -> Optional[np.ndarray]:
        if not self._valid_band(band) or len(self._rows[band]) == 0:
            return None
        if len(self._rows[band]) == 1:
            return self._rows[band][0].reshape(1, -1)
        return np.vstack(self._rows[band])

    def _get_image_uniform_time(
        self,
        band: int,
        target_hz: float,
        *,
        orientation: str | None = None,
        method: str | None = None,
    ) -> Optional[np.ndarray]:
        if not self._valid_band(band) or len(self._rows[band]) == 0:
            return None
        rows_list = list(self._rows[band])
        times_list = list(self._times[band])
        rows = np.vstack(rows_list).astype(np.float32)
        ts = np.array(times_list, dtype=np.float64)
        N, W = rows.shape
        if N == 1 or target_hz <= 0.0:
            xt = rows
        else:
            dt = 1.0 / float(target_hz)
            t0 = float(ts[0])
            t1 = float(ts[-1])
            if t1 <= t0:
                xt = rows
            else:
                n_out = max(2, int(np.floor((t1 - t0) / dt)) + 1)
                t_uni = t0 + np.arange(n_out, dtype=np.float64) * dt
                use_method = (method or self._resample_method or "nearest").lower()
                if use_method == "linear" and N >= 2:
                    xt = np.empty((n_out, W), dtype=np.float32)
                    for col in range(W):
                        xt[:, col] = np.interp(t_uni, ts, rows[:, col])
                else:
                    idx = np.searchsorted(ts, t_uni, side="left")
                    idx = np.clip(idx, 0, N - 1)
                    left = np.clip(idx - 1, 0, N - 1)
                    choose_left = (idx > 0) & ((t_uni - ts[left]) <= (ts[idx] - t_uni))
                    nearest = np.where(choose_left, left, idx)
                    xt = rows[nearest, :]
        xt = np.clip(np.rint(xt), 0, 255).astype(np.uint8)
        orient = (orientation or self._time_axis).lower()
        if orient not in ("x", "y"):
            orient = self._time_axis
        return xt.T if orient == "x" else xt

    @staticmethod
    def _coerce_binary_1d(line: np.ndarray) -> np.ndarray:
        li = np.asarray(line)
        if li.ndim != 1:
            li = li.reshape(-1)
        if li.dtype != np.uint8:
            li = li.astype(np.uint8)
        if li.size and li.max() > 1:
            li = (li > 0).astype(np.uint8)
        return li

    def _trim_band(self, band: int) -> None:
        if len(self._times[band]) == 0:
            return
        newest = float(self._times[band][-1])
        while (
            len(self._times[band]) > 1
            and (newest - float(self._times[band][0])) > self.window_s
        ):
            self._times[band].popleft()
            self._rows[band].popleft()

    def _valid_band(self, band: int) -> bool:
        return 0 <= band < len(self._rows)


def main(capture_backend: Optional[CaptureBackend] = None,
         input_backend: Optional[InputBackend] = None,
         target_fps: int = 120,
         is_desktop: bool = True,
         show_display: bool = True,
         display_fps: int = 60):
    MOVE_VY_THRESH = 0.40
    PHASE_BLEND = 0.85
    PHASE_EMA = 0.20
    STEP_DUR_INIT = 0.30
    SHOW_HLINES = True
    CHAR_BGR_DEFAULT = (92, 172, 255)
    CHAR_TOL_DEFAULT = 4
    SHOW_CHAR_DEBUG = False
    CHAR_DEBUG_WIN = "Character Debug"
    offset_log: list[dict] = []
    angle_deg = 14.5 if is_desktop else 0.0
    SHOW_PLANNER_VIZ = True
    if capture_backend is None:
        cfg = CaptureConfig(monitor=1, region=(0, 27, 247, 449))
        capture_backend = DesktopCaptureBackend(cfg)
    if input_backend is None:
        input_backend = DesktopInputBackend()
    cap = capture_backend
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    target_dt = 1.0 / target_fps
    _display_dt = 1.0 / display_fps if display_fps > 0 else 0.0
    _last_display_ts = 0.0
    _windows_positioned = False
    _windows_created = False
    vertical_fraction = 0.6
    offset = 0
    last_moving = None
    lane_h = 22
    margin_px = 1
    offset_n_lines = 5
    frame_i = 0
    M = None
    raw_w = raw_h = None
    param = 12
    param2 = 14.5 if is_desktop else 0.0
    last_ts = time.perf_counter()
    USE_KALMAN = False
    gameover = GameOverDetector(
        hsv_lo=(17, 160, 200),
        hsv_hi=(30, 255, 255),
        min_frac=0.015,
        consec_needed=3,
    )
    state_machine = BotStateMachine(wait_time=0.2, input_backend=input_backend)
    hlines = None
    if not hasattr(main, "_last_offset0"):
        main._last_offset0 = None
    if not hasattr(main, "_dot_positions_cache"):
        main._dot_positions_cache = [None] * 4
    tracked_indices = [i for i in range(11, 15)]
    sim_started = False
    last_offset_sign = -1
    if not hasattr(main, "_prev_occ_lines"):
        main._prev_occ_lines = None
    velocity_estimator = VelocityEstimator(
        window_s=VELOCITY_WINDOW_S,
        time_on_x=True,
        max_horizontal_run_px=5,
        repeat_suppression=2,
        hough_rho=1.0,
        hough_theta=np.pi / 180.0,
        hough_threshold=10,#set threshold here--------------------------------
        hough_min_line_length=1.0,
        hough_max_line_gap=4.0,
        future_projection_s=VELOCITY_FUTURE_PROJECTION_S,
    )
    planner = Planner(cols_per_sec=velocity_estimator._sampling_rate_hz)
    projection_windows: set[str] = set()
    band_class_store: list[LaneType] = []
    while True:
        frame_start = time.perf_counter()
        dt = max(1e-6, frame_start - last_ts)
        inst_fps = 1.0 / dt
        # Rate-limit display updates so X11 over TCP doesn't choke
        _should_display = show_display and (frame_start - _last_display_ts >= _display_dt)
        frame_bgr = cap.next_frame()
        last_ts = frame_start
        raw_h, raw_w = frame_bgr.shape[:2]
        if M is None:
            center = (raw_w // 2, raw_h // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        frame_bgr = GeometryAnalyzer.deskew(frame_bgr, M, out_size=(raw_h, raw_w))
        shear = math.tan(math.radians(float(param2)))
        if abs(shear) > 1e-8:
            cy = raw_h * 0.5
            M_shear = np.array(
                [
                    [1.0, shear, -shear * cy],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
            frame_bgr = cv2.warpAffine(
                frame_bgr,
                M_shear,
                (raw_w, raw_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
        proc_bgr = cv2.resize(
            frame_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST
        )
        triggered, go_frac = gameover.update(proc_bgr)
        if triggered:
            if is_desktop:
                if gw is not None:
                    _wins = gw.getWindowsWithTitle("MSI App Player")
                    win = _wins[0] if _wins else None
            print("GAME OVER")
            band_class_store = []
            tracked_indices = [i for i in range(11, 15)]
            state_machine.reset()
            time.sleep(3)
            continue
        center, bbox, score, dbg = CharacterTools.detect_character(
            proc_bgr, bgr_target=(92, 172, 255), tol=4, debug=True
        )
        if _should_display and SHOW_CHAR_DEBUG and dbg is not None and "montage" in dbg:
            cv2.imshow(CHAR_DEBUG_WIN, dbg["montage"])
        hsv = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2Lab)
        _, gray_den = DensityMaskUtils.gray_density_map_hsv_from(hsv, lab, ksize=11)
        _, blue_den = DensityMaskUtils.blue_density_map_hsv_from(hsv, lab, ksize=11)
        _, green_den = DensityMaskUtils.green_density_map_hsv_from(hsv, lab, ksize=11)
        nonblue_den = 1.0 - blue_den
        maps = {
            "hsv": hsv,
            "lab": lab,
            "gray_den": gray_den,
            "blue_den": blue_den,
            "green_den": green_den,
            "nonblue_den": nonblue_den,
        }
        cov_g_row, cov_w_row, cov_r_row = BandClassifier.precompute_row_cov(
            maps, margin_px=1, tau_g=0.55, tau_w=0.55, tau_r=0.55, sample_frac=BAND_COV_SAMPLE_FRAC
        )
        cov_rows = np.stack((cov_g_row, cov_w_row, cov_r_row), axis=1)
        offset, offset_meta = GeometryAnalyzer.best_offset_by_line_similarity(
            maps=maps,
            frame_h=proc_bgr.shape[0],
            lane_h=lane_h,
            margin_px=margin_px,
            sim_metric="cosine",
            tau_g=0.55,
            tau_w=0.55,
            tau_r=0.55,
            min_cov_for_label=0.18,
            pair_weights=(0.4, 0.4, 0.2),
            n_lines=offset_n_lines,
            sample_frac=BAND_COV_SAMPLE_FRAC,
            cov_rows=cov_rows,
            agreement_thresh=BAND_AGREEMENT_THRESH,
            prev_offset=main._last_offset0 if main._last_offset0 is not None else None,
            search_radius=OFFSET_SEARCH_RADIUS,
        )
        main._last_offset0 = int(offset)
        # offset_log.append(
        #     {
        #         "frame": frame_i,
        #         "offset": int(offset),
        #         "best_score": offset_meta.get("best_score"),
        #     }
        # )
        offset_sign = round((offset - 11.5) / 11.5)
        band_shift_requested = False
        if offset_sign == -1 and last_offset_sign == 1:
            band_shift_requested = True
            occ_stab = getattr(main, "_occ_stab", None)
            if velocity_estimator is not None:
                velocity_estimator.shift_bands_down_by_one()
            # cached = getattr(main, "_dot_positions_cache", None)
            # if isinstance(cached, list):
            #     main._dot_positions_cache = cached[1:] + [None]
            # tracked_indices = [idx + 1 for idx in tracked_indices]
        last_offset_sign = offset_sign
        H, W = proc_bgr.shape[:2]
        bands = GeometryAnalyzer.find_uniform_bands_with_offset(
            H, lane_h, offset, margin_px=margin_px
        )
        need_full_reclass = (
            not band_class_store or len(band_class_store) != len(bands)
        )
        if need_full_reclass:
            band_class_store = BandClassifier.classify_bands_from_rowcov(
                bands,
                cov_g_row=cov_g_row,
                cov_w_row=cov_w_row,
                cov_r_row=cov_r_row,
                margin_px=1,
                n_lines=offset_n_lines,
                mode="mean",
                min_cov_for_label=0.18,
            )
        elif band_shift_requested and bands:
            prev_labels = band_class_store.copy()
            for idx in range(len(prev_labels) - 1, 0, -1):
                band_class_store[idx] = prev_labels[idx - 1]
            band_zero_label = BandClassifier.classify_bands_from_rowcov(
                [bands[0]],
                cov_g_row=cov_g_row,
                cov_w_row=cov_w_row,
                cov_r_row=cov_r_row,
                margin_px=1,
                n_lines=offset_n_lines,
                mode="mean",
                min_cov_for_label=0.18,
            )[0]
            band_class_store[0] = band_zero_label
        lane_types = band_class_store
        planner.update_lane_types(lane_types)
        char_band_idx: Optional[int] = None
        char_lane_type: Optional[LaneType] = None
        if center is not None and bbox is not None:
            y_exact = center[1]
            char_band_idx = GeometryAnalyzer.band_index_for_y_lower(
                y=y_exact, lane_h=lane_h, offset=offset, frame_h=H, n_bands=len(bands)
            )
            if char_band_idx is not None:
                char_lane_type = lane_types[char_band_idx]
        planner.update_character_band(char_band_idx, char_lane_type)
        frame_i += 1
        disp = proc_bgr.copy()
        h, w = disp.shape[:2]
        character_rect_x: Optional[tuple[int, int]] = None
        if center is not None and char_band_idx is not None and 0 <= char_band_idx < len(bands):
            half_side = 10  # half of 21 rounded down keeps width 21 inclusive
            band_y0, band_y1 = bands[char_band_idx]
            # clamp square to band so it does not bleed into adjacent lanes
            y0_bound = max(0, int(band_y0))
            y1_bound = min(h - 1, int(band_y1 - 1))
            x0 = int(np.clip(center[0] - half_side, 0, w - 1))
            x1 = int(np.clip(center[0] + half_side, 0, w - 1))
            y0 = int(np.clip(center[1] - half_side, y0_bound, y1_bound))
            y1 = int(np.clip(center[1] + half_side, y0_bound, y1_bound))
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 1)
            character_rect_x = (x0, x1)
        cx, cy = w * 0.5, h * 0.5
        theta = math.radians(float(-5.9))
        vx = math.sin(theta)
        vy = math.cos(theta)
        length = max(h, w)
        x0 = int(np.clip(cx - vx * length, 0, w - 1))
        y0 = int(np.clip(cy - vy * length, 0, h - 1))
        x1 = int(np.clip(cx + vx * length, 0, w - 1))
        y1 = int(np.clip(cy + vy * length, 0, h - 1))
        cv2.line(disp, (x0, y0), (x1, y1), (255, 0, 255), 1, cv2.LINE_AA)
        ref_x = float(np.clip(103.69 - 40, 0, w - 1))
        cv2.line(
            disp,
            (int(round(ref_x)), 0),
            (int(round(ref_x)), h - 1),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        denom = (y1 - y0)
        if abs(denom) < 1e-6:
            line_x_from_y = None
        else:
            line_x_from_y = lambda y_val: x0 + (y_val - y0) * (x1 - x0) / denom
        dot_positions: list[tuple[int, float]] = []
        lowest_dot_x = None
        if line_x_from_y is not None:
            band_half = lane_h * 0.5
            for idx in tracked_indices:
                band_top = float(bands[idx][0])
                inter_x = line_x_from_y(band_top)
                inter_x = float(np.clip(inter_x, 0, w - 1))
                dot_y = float(band_top + band_half)
                dot_y = float(np.clip(dot_y, 0, h - 1))
                dot_positions.append((idx, inter_x))
                if lowest_dot_x is None:
                    lowest_dot_x = inter_x
                cv2.circle(
                    disp,
                    (int(round(inter_x)), int(round(dot_y))),
                    6,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )
        top_bottom_line_labels = []
        for i, b in enumerate(bands):
            tb = GeometryAnalyzer.classify_band_top_bottom_lines(
                b,
                maps,
                margin_px=margin_px,
                tau_g=0.55,
                tau_w=0.55,
                tau_r=0.55,
                min_cov=0.18,
                debug=True,
            )
            top_bottom_line_labels.append(tb)
        Visualizer.draw_lane_bands(
            disp,
            [
                Lane(
                    idx=i,
                    y0=y0,
                    y1=y1,
                    height=(y1 - y0),
                    lane_type=lt,
                    mean_rgb=(0, 0, 0),
                    confidence=1.0,
                )
                for i, ((y0, y1), lt) in enumerate(zip(bands, lane_types))
            ],
            show_labels=True,
        )
        free_mask = DensityMaskUtils.free_mask_gray(
            proc_bgr,
            bands,
            lane_types,
            gray_kernel=11,
            gray_tau=0.55,
            min_vert_frac=0.75,
            consider_bottom_frac=0.75,
            # moving_edges=last_moving,
            gap_fill_px=21,
            maps=maps,
            erode_px=0,
        )
        occ_lines = GeometryAnalyzer.build_occ_lines_for_bands(
            free_mask, bands, lane_h, use_bottom_slice=True
        )
        shift_px = 0
    
        push_time = time.perf_counter()
        velocity_estimator.push_occ_lines(occ_lines, t_now=push_time)
        planner.push_tracked_occ_lines(
            occ_lines=occ_lines, tracked_indices=tracked_indices, t_now=push_time
        )

        hough_lines_11_15 = []
        hough_visual_dims_11_15: list[Optional[tuple[int, int]]] = []
        segment_infos: list[Optional[list[dict]]] = []
        for idx in tracked_indices:
            _hough_lines, hough_vis, segment_info = velocity_estimator.probabilistic_hough(
                band=idx,
                orientation="x",
                x_scale=1,
                y_scale=1,
                return_visual=True,
            )
            hough_lines_11_15.append(_hough_lines)
            if hough_vis is not None:
                h, w = hough_vis.shape[:2]
                hough_visual_dims_11_15.append((w, h))
            else:
                hough_visual_dims_11_15.append(None)
            segment_infos.append(segment_info)
            sobel_vis = velocity_estimator.get_latest_sobel_visual(idx)
            # actual_hough_vis = velocity_estimator.get_latest_actual_hough_visual(param)
            # if actual_hough_vis is not None:
            #     cv2.namedWindow("Velocity XT HoughP Actual", cv2.WINDOW_NORMAL)
            #     cv2.imshow("Velocity XT HoughP Actual", actual_hough_vis)
        planner.update_hough_lines(hough_lines_11_15)
        projections, proj_vis_list = planner.project_segments(
            hough_lines=hough_lines_11_15,
            segment_locations=segment_infos,
            hough_visual_dims=hough_visual_dims_11_15,
            character_rect_x=character_rect_x,
            future_time_s=VELOCITY_FUTURE_PROJECTION_S,
            visualize=True,
        )
        if False and _should_display:
            for idx, proj_vis in enumerate(proj_vis_list):
                win_name = f"SP Band {idx}"
                if proj_vis is None:
                    continue
                if win_name not in projection_windows:
                    projection_windows.add(win_name)
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.imshow(win_name, proj_vis)

        # move_cmd = planner.plan_move()
        move_cmd = 'wait'

        if not triggered:
            state_machine.update(move_cmd=move_cmd)
        hud_info = {
            "fps": f"{inst_fps:.1f}",
            "param": param,
            "param2": param2
        }
        Visualizer.draw_hud(disp, hud_info)
        # occ_panel_now = Visualizer.make_occ_lines(
        #     occ_lines_for_display, col_scale=1, band_h=18, show_labels=True, label_w=100
        # )
        occ_panel_now = Visualizer.make_occ_lines(
            occ_lines, col_scale=1, band_h=18, show_labels=True, label_w=100
        )
        main._prev_occ_lines = [l.copy() if l is not None else None for l in occ_lines]
        if _should_display:
            _last_display_ts = frame_start
            if not _windows_created:
                cv2.namedWindow("CrossyBot", cv2.WINDOW_NORMAL)
                _windows_created = True
            cv2.imshow("CrossyBot", disp)
            if not _windows_positioned:
                cv2.moveWindow("CrossyBot", 650, 0)
                _windows_positioned = True
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                cv2.destroyAllWindows()
                return
        if key == ord("]"):
            param += 1
            # param2 += 0.1
        if key == ord("["):
            param -= 1
            # param2 -= 0.1
        work_time = time.perf_counter() - frame_start
        sleep_s = max(0.0, target_dt - work_time)
        if sleep_s > 0:
            time.sleep(sleep_s)
            work_time = target_dt
        inst_fps = 1.0 / work_time

def _parse_runtime_args():
    parser = argparse.ArgumentParser(description="CrossyBot runner")
    parser.add_argument(
        "--backend", choices=["desktop", "adb", "pipe"], default="desktop",
        help="Capture/input backend (default: desktop)",
    )
    parser.add_argument(
        "--serial", default=None,
        help="ADB device serial (passed to adb -s). Only used with --backend adb.",
    )
    parser.add_argument(
        "--fps", type=int, default=120,
        help="Target FPS for the main loop (default: 120)",
    )
    parser.add_argument(
        "--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
        help="Crop region (x y w h) applied to ADB captures. Ignored in desktop mode.",
    )
    parser.add_argument(
        "--no-display", action="store_true", default=False,
        help="Disable all cv2 debug windows (faster — no macOS Cocoa overhead).",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be a positive integer")
    return args

def _build_backends(args) -> Tuple[CaptureBackend, InputBackend]:
    crop = tuple(args.crop) if args.crop else None
    if args.backend == "adb":
        return AdbCaptureBackend(serial=args.serial, crop=crop), AdbInputBackend(serial=args.serial)
    if args.backend == "pipe":
        return PipeCaptureBackend(serial=args.serial, crop=crop), AdbInputBackend(serial=args.serial)
    return DesktopCaptureBackend(CaptureConfig(monitor=1, region=(0, 27, 247, 449))), DesktopInputBackend()

if __name__ == "__main__":
    args = _parse_runtime_args()
    _capture, _input = _build_backends(args)
    main(capture_backend=_capture, input_backend=_input, target_fps=args.fps,
         is_desktop=(args.backend == "desktop"),
         show_display=not args.no_display,
    )
