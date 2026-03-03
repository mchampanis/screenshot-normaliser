# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python>=4.8",
#     "Pillow>=10.0",
#     "numpy>=1.24",
# ]
# ///
"""
Screenshot Normaliser
---------------------
Detects the application window in each screenshot and crops / pads all images
to consistent dimensions with a uniform wallpaper border (using the existing
background in the screenshots).

Works best if you purposefully capture extra wallpaper all around the app in
each screenshot.

Ideally this program is used to normalise a collection of screenshots of the
same app (with same size app window), its primary purpose is to fix the uneven
wallpaper borders you get whilst taking a screenshot free-hand with the snip
tool.

Usage (inline script - no setup needed):
    uv run normaliser.py

Usage (project venv):
    uv sync
    uv run python normaliser.py
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk


# ─────────────────────────────────────────────────────────────────────────────
# DPI / theme helpers  (called before Tk() is created)
# ─────────────────────────────────────────────────────────────────────────────

def _fix_dpi() -> None:
    """
    Declare per-monitor DPI awareness on Windows so the OS does not
    bitmap-scale (blur) the window on high-DPI / 4K displays.
    Must be called before any Tk window is created.
    """
    if sys.platform != "win32":
        return
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(2)   # per-monitor DPI aware v2
    except Exception:
        try:
            from ctypes import windll
            windll.user32.SetProcessDPIAware()    # Vista-era fallback
        except Exception:
            pass


def _os_wants_dark() -> bool:
    """Return True if the OS / desktop environment is set to dark mode."""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            winreg.CloseKey(key)
            return value == 0   # 0 = dark, 1 = light
        except Exception:
            return False
    elif sys.platform == "darwin":
        try:
            import subprocess
            r = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True,
            )
            return r.stdout.strip().lower() == "dark"
        except Exception:
            return False
    else:   # Linux / BSD
        try:
            import subprocess
            r = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                capture_output=True, text=True,
            )
            if "dark" in r.stdout.lower():
                return True
        except Exception:
            pass
        return "dark" in os.environ.get("GTK_THEME", "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def area(self) -> int:
        return self.w * self.h

    def __str__(self) -> str:
        return f"{self.w}x{self.h}  at  ({self.x}, {self.y})"


@dataclass
class ScreenshotEntry:
    path: Path
    image: Optional[np.ndarray] = None   # BGR, loaded on demand
    window: Optional[Rect] = None
    status: str = "pending"              # pending | detected | failed
    error: Optional[str] = None

    @property
    def filename(self) -> str:
        return self.path.name


# ─────────────────────────────────────────────────────────────────────────────
# Window detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_window(img: np.ndarray, sensitivity: float = 0.5) -> Optional[Rect]:
    """
    Detect the dominant rectangular app window in a screenshot.

    Tries three complementary methods in order of reliability:
      1. Luminance threshold — excellent for dark window on bright wallpaper
         (or vice versa); uses Otsu's threshold + row/col projection.
      2. Background colour distance — good when wallpaper has a distinctive
         colour sampled from the image corners.
      3. Hough line detection — fallback for windows with a clear frame border.

    sensitivity: 0.0 = strict -> 1.0 = loose
    """
    h, w = img.shape[:2]
    min_area = 0.10 * w * h

    for method in (_detect_by_luminance, _detect_by_background, _detect_by_edges):
        result = method(img, sensitivity)
        if result and result.area() > min_area and result.w < w and result.h < h:
            return result

    return None


def _longest_window_run(frac: np.ndarray, threshold: float) -> Optional[tuple[int, int]]:
    """
    Find the longest contiguous run of values above `threshold`.
    Returns (start, end) indices, or None.

    Using the longest run (rather than first/last index) avoids being misled
    by stray edge-pixels at the image border, which can spike frac to 1.0.
    """
    mask        = frac > threshold
    transitions = np.diff(mask.astype(np.int8))
    starts      = list(np.where(transitions == 1)[0] + 1)
    ends        = list(np.where(transitions == -1)[0])
    if mask[0]:  starts = [0] + starts  # noqa: E701
    if mask[-1]: ends   = ends + [len(frac) - 1]  # noqa: E701
    if not starts:
        return None
    lengths = [e - s for s, e in zip(starts, ends)]
    i       = int(np.argmax(lengths))
    return int(starts[i]), int(ends[i])


def _detect_by_luminance(img: np.ndarray, sensitivity: float) -> Optional[Rect]:
    """
    Detect window using Otsu's grayscale threshold + row/col projection.

    Determines whether the window is darker or brighter than the wallpaper by
    sampling the four image corners, then projects the foreground mask onto
    rows and columns and finds the largest contiguous band in each direction.
    """
    h, w = img.shape[:2]
    cs   = max(8, min(h, w) // 30)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corner_brightness = float(np.mean([
        gray[:cs, :cs].mean(), gray[:cs, w-cs:].mean(),
        gray[h-cs:, :cs].mean(), gray[h-cs:, w-cs:].mean(),
    ]))

    thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if corner_brightness > thresh_val:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    fg          = binary > 0
    row_frac    = fg.mean(axis=1)
    col_frac    = fg.mean(axis=0)
    frac_thresh = 0.30 - sensitivity * 0.10   # 0.30 strict ... 0.20 loose

    rows = _longest_window_run(row_frac, frac_thresh)
    cols = _longest_window_run(col_frac, frac_thresh)
    if rows is None or cols is None:
        return None

    top, bottom = rows
    left, right = cols
    return Rect(left, top, right - left + 1, bottom - top + 1)


def _detect_by_background(img: np.ndarray, sensitivity: float) -> Optional[Rect]:
    """
    Detect window by comparing each pixel to the colour sampled from the four
    image corners.  Works well when the wallpaper has a distinctive, relatively
    uniform hue that is clearly different from the window content.
    """
    h, w = img.shape[:2]
    cs   = max(8, min(h, w) // 30)

    corner_patches = [
        img[:cs, :cs], img[:cs, w-cs:],
        img[h-cs:, :cs], img[h-cs:, w-cs:],
    ]
    corner_means = np.array([p.mean(axis=(0, 1)) for p in corner_patches], dtype=np.float32)

    if corner_means.std(axis=0).mean() > 40:
        return None

    bg          = np.median(corner_means, axis=0)
    threshold   = 20 + sensitivity * 25
    diff        = np.abs(img.astype(np.float32) - bg).mean(axis=2)
    fg          = diff > threshold
    row_frac    = fg.mean(axis=1)
    col_frac    = fg.mean(axis=0)
    frac_thresh = 0.30 - sensitivity * 0.10

    rows = _longest_window_run(row_frac, frac_thresh)
    cols = _longest_window_run(col_frac, frac_thresh)
    if rows is None or cols is None:
        return None

    top, bottom = rows
    left, right = cols
    return Rect(left, top, right - left + 1, bottom - top + 1)


def _detect_by_edges(img: np.ndarray, sensitivity: float) -> Optional[Rect]:
    """
    Detect window frame via Hough line detection.
    Looks for long horizontal and vertical lines that bound the window rectangle.
    Fallback for images where the window has a clearly drawn border/frame.
    """
    h, w    = img.shape[:2]
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    med   = float(np.median(blurred))
    sigma = 0.25 + sensitivity * 0.40
    lo    = max(0,   int(med * (1 - sigma)))
    hi    = min(255, int(med * (1 + sigma)))
    edges = cv2.Canny(blurred, lo, hi)

    min_len = int(min(w, h) * (0.35 - sensitivity * 0.10))
    lines   = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=max(30, min_len // 3),
        minLineLength=min_len,
        maxLineGap=30,
    )
    if lines is None:
        return None

    margin = 4
    h_ys, v_xs = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dy, dx = abs(y2 - y1), abs(x2 - x1)
        if dx > 0 and dy / dx < 0.10:
            mid_y = (y1 + y2) // 2
            if margin < mid_y < h - margin:
                h_ys.append(mid_y)
        elif dy > 0 and dx / dy < 0.10:
            mid_x = (x1 + x2) // 2
            if margin < mid_x < w - margin:
                v_xs.append(mid_x)

    if not h_ys or not v_xs:
        return None

    top, bottom = min(h_ys), max(h_ys)
    left, right = min(v_xs), max(v_xs)
    bw, bh      = right - left + 1, bottom - top + 1

    if bw < 50 or bh < 50:
        return None
    return Rect(left, top, bw, bh)


def make_normalised_canvas(
    img:       np.ndarray,
    det:       Rect,
    target_w:  int,
    target_h:  int,
    padding:   int,
    edge_mode: str = "smear",
    custom_color: Optional[tuple[int, int, int]] = None, # BGR
) -> np.ndarray:
    """
    Build the normalised output image.

    The window is centered within a target_w x target_h area (no resizing)
    and then wrapped in a uniform border of `padding` pixels.
    """
    ih, iw = img.shape[:2]

    # ── Window crop ───────────────────────────────────────────────────────────
    wx1 = max(0, det.x);          wy1 = max(0, det.y)  # noqa: E702
    wx2 = min(iw, det.x + det.w); wy2 = min(ih, det.y + det.h)  # noqa: E702
    window_crop = img[wy1:wy2, wx1:wx2]

    # If the window isn't the target size, it will be centered in a blank canvas
    # and the edges will be filled by the _blit logic below.

    out_w = target_w + 2 * padding
    out_h = target_h + 2 * padding

    # ── Wallpaper border / Fill ───────────────────────────────────────────────
    # Top-Centered Alignment:
    # Use exact specified padding for the top.
    # Add all "normalisation" extra height to the bottom.
    # Center the window horizontally.
    pad_t = padding
    pad_b = out_h - det.h - pad_t
    pad_l = (target_w - det.w) // 2 + padding
    pad_r = out_w - det.w - pad_l

    # ── Fast path ─────────────────────────────────────────────────────────────
    # If the source image already contains enough real wallpaper to satisfy the
    # target size + padding, we can just take a direct crop.
    # (Bypassed if using a custom color override)
    if (edge_mode != "custom" and
            det.y - pad_t >= 0 and det.y + det.h + pad_b <= ih
            and det.x - pad_l >= 0 and det.x + det.w + pad_r <= iw):
        return img[det.y - pad_t : det.y + det.h + pad_b,
                   det.x - pad_l : det.x + det.w + pad_r].copy()

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Source boundaries for the wallpaper (anchored on actual detected window)
    sr_t1, sr_t2 = det.y - pad_t, det.y
    sr_b1, sr_b2 = det.y + det.h, det.y + det.h + pad_b
    sc_l1, sc_l2 = det.x - pad_l, det.x
    sc_r1, sc_r2 = det.x + det.w, det.x + det.w + pad_r

    # Base background color sampled from corners (calculated once for efficiency)
    if edge_mode == "custom" and custom_color is not None:
        bg_color = np.array(custom_color, dtype=np.uint8)
    else:
        cs = max(8, min(ih, iw) // 30)
        corners = [img[:cs, :cs], img[:cs, iw-cs:],
                   img[ih-cs:, :cs], img[ih-cs:, iw-cs:]]
        bg_color = np.median([c.mean(axis=(0, 1)) for c in corners], axis=0).astype(np.uint8)

    def _blit(dst_r: int, dst_c: int, dst_h: int, dst_w: int,
              sr1: int, sr2: int, sc1: int, sc2: int) -> None:
        if dst_h <= 0 or dst_w <= 0:
            return

        csr1 = max(0, sr1); csr2 = min(ih, sr2)  # noqa: E702
        csc1 = max(0, sc1); csc2 = min(iw, sc2)  # noqa: E702

        if csr2 <= csr1 or csc2 <= csc1:
            canvas[dst_r:dst_r+dst_h, dst_c:dst_c+dst_w] = bg_color
            return

        patch = img[csr1:csr2, csc1:csc2].copy()

        # Fill any missing area (where window is too close to image edge)
        if edge_mode == "smear":
            if csr1 > sr1: patch = np.concatenate([np.repeat(patch[:1], csr1-sr1, axis=0), patch], axis=0)  # noqa: E701
            if csr2 < sr2: patch = np.concatenate([patch, np.repeat(patch[-1:], sr2-csr2, axis=0)], axis=0)  # noqa: E701
            if csc1 > sc1: patch = np.concatenate([np.repeat(patch[:, :1], csc1-sc1, axis=1), patch], axis=1)  # noqa: E701
            if csc2 < sc2: patch = np.concatenate([patch, np.repeat(patch[:, -1:], sc2-csc2, axis=1)], axis=1)  # noqa: E701
        else: # median mode
            target_patch = np.full((sr2 - sr1, sc2 - sc1, 3), bg_color, dtype=np.uint8)
            target_patch[csr1-sr1 : csr2-sr1, csc1-sc1 : csc2-sc1] = patch
            patch = target_patch

        if patch.shape[0] != dst_h or patch.shape[1] != dst_w:
            patch = cv2.resize(patch, (dst_w, dst_h), interpolation=cv2.INTER_LANCZOS4)
        canvas[dst_r:dst_r+dst_h, dst_c:dst_c+dst_w] = patch

    # 4 corners
    _blit(0,             0,             pad_t, pad_l, sr_t1, sr_t2, sc_l1, sc_l2)
    _blit(0,             pad_l + det.w, pad_t, pad_r, sr_t1, sr_t2, sc_r1, sc_r2)
    _blit(pad_t + det.h, 0,             pad_b, pad_l, sr_b1, sr_b2, sc_l1, sc_l2)
    _blit(pad_t + det.h, pad_l + det.w, pad_b, pad_r, sr_b1, sr_b2, sc_r1, sc_r2)

    # 4 edge strips
    _blit(0,             pad_l,         pad_t, det.w, sr_t1, sr_t2, det.x, det.x + det.w)
    _blit(pad_t + det.h, pad_l,         pad_b, det.w, sr_b1, sr_b2, det.x, det.x + det.w)
    _blit(pad_t,         0,             det.h, pad_l, det.y, det.y + det.h, sc_l1, sc_l2)
    _blit(pad_t,         pad_l + det.w, det.h, pad_r, det.y, det.y + det.h, sc_r1, sc_r2)

    # ── Window composite ──────────────────────────────────────────────────────
    canvas[pad_t:pad_t + det.h, pad_l:pad_l + det.w] = window_crop
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _unique_path(path: Path) -> Path:
    """Return path unchanged if it does not exist, otherwise append _2, _3, … to the stem."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    for i in range(2, 10_000):
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    return path


def normalise_all(
    entries:     list[ScreenshotEntry],
    padding:     int,
    output_dir:  Path,
    progress_cb  = None,
    edge_mode:   str = "smear",
    custom_color: Optional[tuple[int, int, int]] = None, # BGR
) -> tuple[list[str], list[str]]:
    """
    Pad each entry to the maximum dimensions of all detected windows
    to avoid resizing, add a border, and save.
    """
    valid = [e for e in entries if e.window is not None]
    if not valid:
        return [], ["No successful detections to normalise."]

    # Use maximums to ensure all windows fit without resizing
    target_w = max(e.window.w for e in valid if e.window is not None)
    target_h = max(e.window.h for e in valid if e.window is not None)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved:  list[str] = []
    errors: list[str] = []

    for i, entry in enumerate(entries):
        if progress_cb:
            progress_cb(i, len(entries), entry.filename)

        if entry.image is None or entry.window is None:
            # Re-load if it was cleared for memory
            entry.image = cv2.imread(str(entry.path))
            if entry.image is None or entry.window is None:
                errors.append(f"{entry.filename}: skipped -- could not load image")
                continue

        det    = entry.window
        img    = entry.image
        canvas = make_normalised_canvas(img, det, target_w, target_h, padding, edge_mode, custom_color)

        suffix   = entry.path.suffix.lower() or ".png"
        out_path = _unique_path(output_dir / f"{entry.path.stem}_normalised{suffix}")
        try:
            cv2.imwrite(str(out_path), canvas)
            saved.append(str(out_path))
        except Exception as exc:
            errors.append(f"{entry.filename}: {exc}")

    if progress_cb:
        progress_cb(len(entries), len(entries), "Done")

    return saved, errors


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_ICON = {"pending": "○", "detected": "✓", "failed": "✗"}


class App(tk.Tk):

    # ── Colour palettes ───────────────────────────────────────────────────────

    _LIGHT: dict = dict(
        bg        = "#f5f5f5",
        bg_alt    = "#ffffff",
        bg_input  = "#ffffff",
        fg        = "#1e1e1e",
        fg_dim    = "#666666",
        border    = "#cccccc",
        sel_bg    = "#0078d4",
        sel_fg    = "#ffffff",
        btn_bg    = "#e1e1e1",
        btn_hover = "#c8c8c8",
        canvas_bg = "#d8d8d8",
        ok_fg     = "#107c10",
        err_fg    = "#c42b1c",
        list_ok   = "#dff0d8",
        list_err  = "#fdd8d8",
        list_pend = "#ffffff",
    )

    _DARK: dict = dict(
        bg        = "#1e1e1e",
        bg_alt    = "#252526",
        bg_input  = "#3c3c3c",
        fg        = "#d4d4d4",
        fg_dim    = "#888888",
        border    = "#3f3f46",
        sel_bg    = "#094771",
        sel_fg    = "#ffffff",
        btn_bg    = "#3c3c3c",
        btn_hover = "#505050",
        canvas_bg = "#1a1a2e",
        ok_fg     = "#6ec547",
        err_fg    = "#f14c4c",
        list_ok   = "#1e3a1e",
        list_err  = "#3a1e1e",
        list_pend = "#2d2d2d",
    )

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        super().__init__()
        self.title("Screenshot Normaliser")
        self.geometry("1020x660")
        self.minsize(760, 500)

        self.entries:        list[ScreenshotEntry] = []
        self._sel_idx:       int                   = -1
        self._preview_photo                        = None
        self._work_lock                            = threading.Lock()

        self._dark = _os_wants_dark()
        self._c    = self._DARK if self._dark else self._LIGHT

        self._apply_theme()
        self._build_ui()

        self.bind("<Control-o>", lambda _e: self.add_files())
        self.bind("<Control-O>", lambda _e: self.add_files())

        # Title-bar colour (Windows 11) needs the HWND, available after mapping
        self.after(50, self._set_titlebar_theme)

    # ── Theme application ─────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        c = self._c
        self.configure(bg=c["bg"])

        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".",
            background=c["bg"], foreground=c["fg"],
            bordercolor=c["border"], darkcolor=c["bg"], lightcolor=c["bg"],
            troughcolor=c["bg_input"], focuscolor=c["sel_bg"],
        )
        style.configure("TFrame",        background=c["bg"])
        style.configure("TLabel",        background=c["bg"],      foreground=c["fg"])
        style.configure("TSeparator",    background=c["border"])
        style.configure("TPanedWindow",  background=c["border"])

        style.configure("TButton",
            background=c["btn_bg"], foreground=c["fg"],
            bordercolor=c["border"], relief="flat", padding=(8, 4),
        )
        style.map("TButton",
            background=[("active", c["btn_hover"]), ("pressed", c["border"])],
            foreground=[("disabled", c["fg_dim"])],
            relief=[("pressed", "flat")],
        )
        style.configure("TEntry",
            fieldbackground=c["bg_input"], foreground=c["fg"],
            bordercolor=c["border"], insertcolor=c["fg"],
        )
        style.configure("TSpinbox",
            fieldbackground=c["bg_input"], foreground=c["fg"],
            bordercolor=c["border"], arrowcolor=c["fg"],
        )
        style.configure("TScale",
            background=c["bg"], troughcolor=c["bg_input"], sliderthickness=14,
        )
        style.configure("TProgressbar",
            troughcolor=c["bg_alt"], background="#0078d4", bordercolor=c["border"],
        )
        style.configure("TScrollbar",
            background=c["bg_alt"], troughcolor=c["bg"],
            arrowcolor=c["fg"], bordercolor=c["border"],
        )

    def _set_titlebar_theme(self) -> None:
        """Paint the Windows 11 title bar dark/light to match the app theme."""
        if sys.platform != "win32":
            return
        try:
            from ctypes import windll, byref, c_int, sizeof
            # Use 20 for Windows 11 / later Win10, 19 for older Win10
            # DWMWA_USE_IMMERSIVE_DARK_MODE
            hwnd  = windll.user32.GetParent(self.winfo_id())
            value = c_int(1 if self._dark else 0)
            for attr in (20, 19):
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, byref(value), sizeof(value)
                )
        except Exception:
            pass

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        c = self._c

        # Pack order matters: BOTTOM items first, then the expanding centre pane.

        # Status bar
        self.status_var = tk.StringVar(value="Ready -- add screenshots to begin.")
        ttk.Label(
            self, textvariable=self.status_var,
            relief=tk.SUNKEN, anchor=tk.W, padding=(8, 3),
        ).pack(fill=tk.X, side=tk.BOTTOM)

        # Progress bar
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=(0, 2))

        # Settings / actions bar
        sf = ttk.Frame(self, padding=(8, 5))
        sf.pack(fill=tk.X, side=tk.BOTTOM)

        ttk.Label(sf, text="Padding:").pack(side=tk.LEFT)
        self.padding_var = tk.IntVar(value=20)
        ttk.Spinbox(sf, from_=0, to=500, textvariable=self.padding_var, width=5).pack(
            side=tk.LEFT, padx=(3, 1)
        )
        ttk.Label(sf, text="px").pack(side=tk.LEFT, padx=(0, 14))

        ttk.Label(sf, text="Edge fill:").pack(side=tk.LEFT)
        self.edge_var = tk.StringVar(value="smear")
        cb = ttk.Combobox(sf, textvariable=self.edge_var, values=["smear", "median", "custom"], width=8, state="readonly")
        cb.pack(side=tk.LEFT, padx=(3, 4))

        self.color_btn = ttk.Button(sf, text="Pick Color...", command=self.pick_fill_color)
        self.color_btn.pack(side=tk.LEFT, padx=2)
        self.color_swatch = tk.Canvas(sf, width=20, height=20, bd=1, highlightthickness=0, bg="#ffffff")
        self.color_swatch.pack(side=tk.LEFT, padx=(2, 14))
        self.custom_color_rgb = (255, 255, 255) # Default white

        ttk.Label(sf, text="Detection sensitivity:").pack(side=tk.LEFT)
        self.sens_var = tk.DoubleVar(value=0.5)
        ttk.Scale(
            sf, from_=0, to=1, variable=self.sens_var,
            orient=tk.HORIZONTAL, length=120,
        ).pack(side=tk.LEFT, padx=(3, 2))
        ttk.Label(sf, text="Low -> High", foreground=c["fg_dim"]).pack(
            side=tk.LEFT, padx=(0, 16)
        )

        ttk.Separator(sf, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(sf, text="Detect Windows",  command=self.run_detect   ).pack(side=tk.LEFT, padx=4)
        ttk.Button(sf, text="Normalise & Save", command=self.run_normalise).pack(side=tk.LEFT, padx=4)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, side=tk.BOTTOM)

        # Toolbar
        tb = ttk.Frame(self, padding=(8, 6))
        tb.pack(fill=tk.X, side=tk.TOP)
        ttk.Button(tb, text="+ Add Files", command=self.add_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="Clear All",   command=self.clear_all).pack(side=tk.LEFT, padx=2)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(tb, text="Output folder:").pack(side=tk.LEFT)
        self.out_var = tk.StringVar()
        ttk.Entry(tb, textvariable=self.out_var, width=38).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Button(tb, text="Browse...", command=self.browse_output).pack(side=tk.LEFT)

        # Centre: split pane
        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left: file list
        lf = ttk.Frame(pane, width=230)
        pane.add(lf, weight=0)

        ttk.Label(lf, text="FILES", font=("TkDefaultFont", 9, "bold")).pack(
            anchor=tk.W, padx=4, pady=(2, 1)
        )
        lc = ttk.Frame(lf)
        lc.pack(fill=tk.BOTH, expand=True, padx=4)

        sb = ttk.Scrollbar(lc, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(
            lc, yscrollcommand=sb.set, selectmode=tk.SINGLE,
            font=("TkDefaultFont", 9), activestyle="none",
            relief=tk.FLAT, bd=0,
            bg=c["bg_alt"],  fg=c["fg"],
            selectbackground=c["sel_bg"], selectforeground=c["sel_fg"],
        )
        sb.config(command=self.listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # Right: preview canvas
        rf = ttk.Frame(pane)
        pane.add(rf, weight=1)

        ttk.Label(rf, text="PREVIEW", font=("TkDefaultFont", 9, "bold")).pack(
            anchor=tk.W, padx=4, pady=(2, 1)
        )
        self.canvas = tk.Canvas(rf, bg=c["canvas_bg"], bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4)
        self.canvas.bind("<Configure>", lambda _e: self._redraw_preview())

        self.info_label = ttk.Label(
            rf, text="", foreground=c["fg_dim"], font=("TkFixedFont", 9)
        )
        self.info_label.pack(anchor=tk.W, padx=8, pady=2)

    # ── File management ───────────────────────────────────────────────────────

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select screenshots",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        added = 0
        for p in paths:
            path = Path(p)
            if not any(e.path == path for e in self.entries):
                self.entries.append(ScreenshotEntry(path=path))
                added += 1

        if added:
            self._refresh_list()
            if not self.out_var.get():
                self.out_var.set(str(Path(paths[0]).parent / "normalised"))
            n = len(self.entries)
            self.status_var.set(
                f"{n} file(s) loaded.  Click 'Detect Windows' to analyse them."
            )

    def clear_all(self) -> None:
        self.entries.clear()
        self._sel_idx = -1
        self._refresh_list()
        self.canvas.delete("all")
        self.info_label.config(text="")
        self.status_var.set("Ready -- add screenshots to begin.")

    def browse_output(self) -> None:
        d = filedialog.askdirectory(title="Choose output folder")
        if d:
            self.out_var.set(d)

    def pick_fill_color(self) -> None:
        rgb, hex_color = colorchooser.askcolor(
            initialcolor="#ffffff", title="Choose fill color"
        )
        if rgb and hex_color:
            self.custom_color_rgb = tuple(int(x) for x in rgb)
            self.color_swatch.config(bg=hex_color)
            self.edge_var.set("custom")

    # ── List helpers ──────────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        c = self._c
        self.listbox.delete(0, tk.END)
        for entry in self.entries:
            icon = _STATUS_ICON.get(entry.status, "○")
            self.listbox.insert(tk.END, f"  {icon}  {entry.filename}")
        status_bg = {
            "pending":  c["list_pend"],
            "detected": c["list_ok"],
            "failed":   c["list_err"],
        }
        for i, entry in enumerate(self.entries):
            self.listbox.itemconfig(
                i, background=status_bg.get(entry.status, c["list_pend"])
            )

    def _on_select(self, _event) -> None:
        sel = self.listbox.curselection()
        if sel:
            self._sel_idx = sel[0]
            self._redraw_preview()

    # ── Preview ───────────────────────────────────────────────────────────────

    def _redraw_preview(self) -> None:
        self.canvas.delete("all")
        self.info_label.config(text="")

        idx = self._sel_idx
        if idx < 0 or idx >= len(self.entries):
            return
        entry = self.entries[idx]

        if entry.image is None:
            self.info_label.config(text="Loading\u2026", foreground=self._c["fg_dim"])
            threading.Thread(
                target=self._preview_load_worker, args=(idx,), daemon=True
            ).start()
            return

        self._draw_preview(idx)

    def _preview_load_worker(self, idx: int) -> None:
        entry = self.entries[idx]
        img = cv2.imread(str(entry.path))
        if self._sel_idx != idx:
            return  # Selection changed while loading
        if img is not None:
            entry.image = img
        self.after(0, lambda: self._on_preview_loaded(idx))

    def _on_preview_loaded(self, idx: int) -> None:
        if self._sel_idx != idx:
            return
        entry = self.entries[idx]
        if entry.image is None:
            self.info_label.config(
                text=f"Cannot load {entry.filename}", foreground=self._c["err_fg"]
            )
            return
        if self._sel_idx == idx:
            self._draw_preview(idx)

    def _draw_preview(self, idx: int) -> None:
        entry = self.entries[idx]
        img = entry.image
        if img is None:
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil     = Image.fromarray(img_rgb)

        if entry.window:
            d    = entry.window
            draw = ImageDraw.Draw(pil)
            lw   = max(2, min(d.w, d.h) // 80)
            draw.rectangle([d.x, d.y, d.x + d.w, d.y + d.h], outline=(0, 230, 80), width=lw)
            tick = lw * 8
            for ox, oy, dx, dy in [
                (d.x,       d.y,       +1, +1),
                (d.x + d.w, d.y,       -1, +1),
                (d.x,       d.y + d.h, +1, -1),
                (d.x + d.w, d.y + d.h, -1, -1),
            ]:
                draw.line([(ox, oy), (ox + dx * tick, oy            )], fill=(0, 230, 80), width=lw)
                draw.line([(ox, oy), (ox,              oy + dy * tick)], fill=(0, 230, 80), width=lw)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        pil.thumbnail((cw - 4, ch - 4), Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(pil)
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._preview_photo)

        if entry.window:
            w = entry.window
            self.info_label.config(
                text=f"Detected window: {w.w}x{w.h}  at  ({w.x}, {w.y})",
                foreground=self._c["ok_fg"],
            )
        elif entry.status == "failed":
            self.info_label.config(
                text=f"Detection failed -- try raising sensitivity.  {entry.error or ''}",
                foreground=self._c["err_fg"],
            )
        else:
            self.info_label.config(text="Not yet analysed.", foreground=self._c["fg_dim"])

    # ── Detection ─────────────────────────────────────────────────────────────

    def run_detect(self) -> None:
        if not self.entries:
            messagebox.showwarning("No files", "Add screenshots first.")
            return
        if not self._work_lock.acquire(blocking=False):
            return
        threading.Thread(target=self._detect_worker, daemon=True).start()

    def _detect_worker(self) -> None:
        try:
            sensitivity = self.sens_var.get()
            total       = len(self.entries)
            self.after(0, lambda: self.progress.configure(maximum=total, value=0))

            for i, entry in enumerate(self.entries):
                self.after(0, lambda fn=entry.filename, j=i+1:
                           self.status_var.set(f"Detecting {fn}  ({j}/{total})..."))

                if entry.image is None:
                    img = cv2.imread(str(entry.path))
                    if img is None:
                        entry.status = "failed"
                        entry.error  = "Could not load image"
                        self.after(0, self._refresh_list)
                        self.after(0, lambda v=i+1: self.progress.configure(value=v))
                        continue
                    entry.image = img

                det = detect_window(entry.image, sensitivity)
                if det:
                    entry.window = det
                    entry.status = "detected"
                    entry.error  = None
                else:
                    entry.window = None
                    entry.status = "failed"
                    entry.error  = "No window boundary found -- try adjusting sensitivity"

                # Keep image in memory ONLY if it's currently being previewed
                if self._sel_idx != i:
                    entry.image = None

                captured_i = i
                self.after(0, self._refresh_list)
                self.after(0, lambda ci=captured_i: self._maybe_redraw(ci))
                self.after(0, lambda v=i+1: self.progress.configure(value=v))

            ok   = sum(1 for e in self.entries if e.status == "detected")
            fail = total - ok
            sizes    = sorted({f"{e.window.w}x{e.window.h}" for e in self.entries if e.window})
            size_str = ("  |  window sizes: " + ", ".join(sizes)) if sizes else ""

            if len(sizes) > 1:
                widths  = [e.window.w for e in self.entries if e.window]
                heights = [e.window.h for e in self.entries if e.window]
                if max(widths) - min(widths) > 20 or max(heights) - min(heights) > 20:
                    size_str += "  -- sizes vary, check detections before saving"

            self.after(0, lambda: self.status_var.set(
                f"Detection complete -- {ok} detected, {fail} failed{size_str}"
            ))
            self.after(0, lambda: self.progress.configure(value=0))
        finally:
            self._work_lock.release()

    def _maybe_redraw(self, idx: int) -> None:
        if self._sel_idx == idx:
            self._redraw_preview()

    # ── Normalise ─────────────────────────────────────────────────────────────

    def run_normalise(self) -> None:
        if not self.entries:
            messagebox.showwarning("No files", "Add screenshots first.")
            return
        if not any(e.window for e in self.entries):
            messagebox.showwarning("No detections", "Run 'Detect Windows' first.")
            return
        out = self.out_var.get().strip()
        if not out:
            messagebox.showwarning("No output folder", "Set an output folder first.")
            return
        try:
            padding = self.padding_var.get()
        except tk.TclError:
            messagebox.showerror("Invalid padding", "Padding must be a whole number ≥ 0.")
            return
        if padding < 0:
            messagebox.showerror("Invalid padding", "Padding must be ≥ 0.")
            return
        if not self._work_lock.acquire(blocking=False):
            return
        edge_mode = self.edge_var.get()
        custom_color = self.custom_color_rgb[::-1] # RGB -> BGR
        threading.Thread(target=self._normalise_worker, args=(padding, edge_mode, custom_color), daemon=True).start()

    def _normalise_worker(self, padding: int, edge_mode: str, custom_color: tuple[int, int, int]) -> None:
        out_dir = Path(self.out_var.get())
        total   = len(self.entries)
        self.after(0, lambda: self.progress.configure(maximum=total, value=0))

        try:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.after(0, lambda e=exc, d=out_dir: self._handle_output_dir_error(e, d))
                self.after(0, lambda: self.progress.configure(value=0))
                return

            def progress_cb(i: int, n: int, name: str) -> None:
                self.after(0, lambda: self.status_var.set(f"Saving {name}  ({i}/{n})..."))
                self.after(0, lambda: self.progress.configure(value=i))

            try:
                saved, errors = normalise_all(self.entries, padding, out_dir, progress_cb, edge_mode, custom_color)
            except Exception as exc:
                self.after(0, lambda e=exc: messagebox.showerror(
                    "Unexpected error", f"Normalisation failed:\n{e}"
                ))
                self.after(0, lambda: self.status_var.set("Normalisation failed -- see error dialog."))
                self.after(0, lambda: self.progress.configure(value=0))
                return

            # Free memory now that images have been saved.
            for entry in self.entries:
                entry.image = None

            self.after(0, lambda: self.progress.configure(value=0))
            msg = f"Saved {len(saved)} image(s) to:\n{out_dir}"
            if errors:
                msg += "\n\nSkipped / errors:\n" + "\n".join(errors)

            self.after(0, lambda: self.status_var.set(f"Done -- {len(saved)} image(s) saved."))
            self.after(0, lambda: messagebox.showinfo("Normalisation complete", msg))
        finally:
            self._work_lock.release()

    def _handle_output_dir_error(self, exc: OSError, attempted: Path) -> None:
        """Called on the main thread when the output directory cannot be created."""
        messagebox.showerror(
            "Cannot create output folder",
            f"Could not create:\n{attempted}\n\n{exc}\n\nPlease choose a different folder.",
        )
        new_dir = filedialog.askdirectory(title="Choose a writable output folder")
        if new_dir:
            self.out_var.set(new_dir)
            self.status_var.set("Output folder updated -- click Normalise & Save to try again.")
        else:
            self.status_var.set("Normalisation cancelled -- no output folder selected.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _fix_dpi()   # must be before Tk() is created
    app = App()
    # Scale fonts to actual screen DPI so text isn't tiny on high-DPI displays
    if sys.platform == "win32":
        try:
            from ctypes import windll
            dpi = windll.user32.GetDpiForSystem()
            app.tk.call("tk", "scaling", dpi / 72)
        except Exception:
            pass
    app.mainloop()


if __name__ == "__main__":
    main()
