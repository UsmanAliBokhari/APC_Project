"""
ui/SkinScanApp.py
----------
Class for SkinScan app UI.
We have a two-column workspace: images left, results right.
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import numpy as np

from models.skin_lesion import SkinLesion
from processing.enhancer import Enhancer
from processing.segmenter import Segmenter
from processing.feature_extractor import FeatureExtractor
from detection.detector import Detector
from utils.helpers import (
    img_to_photoimage,
    image_resize,
    load_image,
)
from ui.theme import *
from ui.widgets import _btn, _divider

class SkinScanApp:
    """Main application window for SkinScan melanoma detection."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self._detector = Detector()
        self._configure_window()
        self._configure_styles()
        self._build_ui()

    def _configure_window(self):
        """Set window title, size, and background colour."""
        self.root.title("SkinScan")
        self.root.geometry("1000x680")
        self.root.minsize(800, 560)
        self.root.configure(bg=C_BG)

    def _configure_styles(self):
        """Configure ttk widget styles and theme."""
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Scan.Horizontal.TProgressbar",
                    troughcolor=C_SURFACE_3, background=C_ACCENT, thickness=3)

    def _build_ui(self):
        """Build the two-column workspace: image left, results right."""
        workspace = tk.Frame(self.root, bg=C_BG)
        workspace.pack(fill="both", expand=True, padx=16, pady=16)
        workspace.columnconfigure(0, weight=1)
        workspace.columnconfigure(1, weight=1)
        workspace.rowconfigure(0, weight=1)

        left = tk.Frame(workspace, bg=C_SURFACE)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        btn_row = tk.Frame(left, bg=C_SURFACE, padx=12, pady=10)
        btn_row.pack(fill="x")

        self._btn_load = _btn(btn_row, "📂  Load", C_SURFACE_3, C_TEXT, self._on_load_image, padx=12)
        self._btn_load.pack(side="left", padx=(0, 8))

        self._btn_analyze = _btn(btn_row, "▶  Analyse", C_ACCENT, C_TEXT, self._on_analyze,
                                 state="disabled", padx=14)
        self._btn_analyze.pack(side="left")

        _divider(left, bg=C_BORDER)

        self._canvas = tk.Canvas(left, bg=C_CANVAS_BG, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=12, pady=12)
        self._canvas.create_text(210, 150, text="No image loaded",
                                 fill=C_TEXT_3, font=FONT_SMALL, tags="placeholder")

        self._progress = ttk.Progressbar(left, mode="indeterminate",
                                         style="Scan.Horizontal.TProgressbar")
        self._photo = None

        right = tk.Frame(workspace, bg=C_SURFACE)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        tk.Label(right, text="Results", font=FONT_HEAD,
                 bg=C_SURFACE, fg=C_TEXT).pack(anchor="w", padx=14, pady=(12, 4))
        _divider(right, bg=C_BORDER)

        summary = tk.Frame(right, bg=C_SURFACE, padx=14, pady=14)
        summary.pack(fill="x")

        self._risk_label = tk.Label(summary, text="—", font=(F_DISP, 22, "bold"),
                                    bg=C_SURFACE, fg=C_TEXT_2, anchor="w")
        self._risk_label.pack(anchor="w")

        self._score_label = tk.Label(summary, text="Run analysis to see results",
                                     font=FONT_BODY, bg=C_SURFACE, fg=C_TEXT_3, anchor="w")
        self._score_label.pack(anchor="w", pady=(4, 0))

        _divider(right, bg=C_BORDER, padx=14)

        bars = tk.Frame(right, bg=C_SURFACE, padx=14, pady=10)
        bars.pack(fill="x")

        self._bar_fills:  dict[str, tuple[tk.Frame, tk.Frame]] = {}
        self._val_labels: dict[str, tk.Label] = {}

        self._ABCDE = [
            ("asymmetry",           "A  Asymmetry",       1.0),
            ("border_irregularity", "B  Border",          1.0),
            ("color_variance",      "C  Colour Variance", 1.0),
            ("diameter_norm",       "D  Diameter",        1.0),
            ("texture_contrast",    "E  Texture",         2.0),
        ]

        for i, (key, label, _) in enumerate(self._ABCDE):
            row_bg = C_SURFACE_3 if i % 2 == 0 else C_SURFACE
            row = tk.Frame(bars, bg=row_bg)
            row.pack(fill="x")

            tk.Label(row, text=label, width=20, anchor="w",
                     font=FONT_SMALL, bg=row_bg, fg=C_TEXT,
                     padx=6, pady=6).pack(side="left")

            trough = tk.Frame(row, bg=C_SURFACE_2, height=6, width=140)
            trough.pack(side="left", padx=(4, 8))
            trough.pack_propagate(False)

            fill = tk.Frame(trough, bg=C_ACCENT, height=6, width=0)
            fill.place(x=0, y=0, relheight=1.0, width=0)
            self._bar_fills[key] = (trough, fill)

            val = tk.Label(row, text="—", width=6, anchor="w",
                           font=FONT_MONO, bg=row_bg, fg=C_TEXT)
            val.pack(side="left")
            self._val_labels[key] = val

        tk.Frame(self.root, bg=C_BORDER, height=1).pack(fill="x", side="bottom")
        sbar = tk.Frame(self.root, bg=C_HEADER, height=26)
        sbar.pack(fill="x", side="bottom")
        sbar.pack_propagate(False)
        self._status_var = tk.StringVar(value="Load an image to begin.")
        tk.Label(sbar, textvariable=self._status_var,
                 font=FONT_SMALL, bg=C_HEADER, fg=C_TEXT_2
                 ).pack(side="left", padx=12, pady=5)

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        """Update the status bar message."""
        self._status_var.set(msg)

    def _set_busy(self, busy: bool):
        """Show or hide the progress bar during long-running tasks."""
        if busy:
            self._progress.pack(fill="x", padx=12, pady=(0, 8))
            self._progress.start(12)
        else:
            self._progress.stop()
            self._progress.pack_forget()

    def _draw_on_canvas(self, img: np.ndarray):
        """Resize img to fit the canvas and draw it centered."""
        cw = self._canvas.winfo_width()  if self._canvas.winfo_width()  > 1 else 460
        ch = self._canvas.winfo_height() if self._canvas.winfo_height() > 1 else 340
        self._photo = img_to_photoimage(img, max_size=(cw, ch))
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, anchor="center", image=self._photo)

    # ------------------------------------------------------------------
    # Callbacks / event handlers
    # ------------------------------------------------------------------

    def _on_load_image(self):
        """Open a file dialog, load the selected image, and display it."""
        path = filedialog.askopenfilename(
            title="Select skin lesion image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("All files", "*.*")])
        if not path:
            return
        img = load_image(path)
        if img is None:
            messagebox.showerror("Load error", f"Could not read image:\n{path}")
            return
        h, w = img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            img = image_resize(img, int(w * scale), int(h * scale))

        lesion = SkinLesion(image_path=path, body_location="Unknown")
        lesion.raw_image = img
        self._current_lesion = lesion

        self._draw_on_canvas(img)
        self._btn_analyze.config(state="normal")
        self._set_status(f"Loaded: {lesion.image_name}  ({w}×{h} px)")

    def _on_analyze(self):
        """Kick off the analysis pipeline in a background thread."""
        if self._current_lesion is None:
            return
        self._set_busy(True)
        self._set_status("Running analysis…")
        threading.Thread(target=self._analysis_worker, daemon=True).start()

    def _analysis_worker(self):
        """Run enhancement, segmentation, and feature extraction off the main thread."""
        try:
            lesion = self._current_lesion
            lesion.enhanced_image = Enhancer(lesion.raw_image).result
            lesion.mask = Segmenter(lesion.enhanced_image).result
            extractor = FeatureExtractor(lesion.enhanced_image, lesion.mask)
            lesion.features = extractor.result
            
            risk_score, risk_level = self._detector.predict(lesion.features)
            lesion.risk_score = risk_score
            lesion.risk_level = risk_level
            self.root.after(0, self._on_analysis_done, lesion, None)
        except Exception as exc:
            self.root.after(0, self._on_analysis_done, None, exc)

    def _on_analysis_done(self, lesion: SkinLesion, error: Exception | None):
        """Continuation: update the UI once analysis completes or fails."""
        self._set_busy(False)
        if error:
            messagebox.showerror("Analysis error", str(error))
            self._set_status("Analysis failed.")
            return

        colour = RISK_COLOURS.get(lesion.risk_level, C_TEXT_2)
        self._risk_label.config(text=f"{lesion.risk_level.upper()} RISK", fg=colour)
        self._score_label.config(text=f"Risk score  {lesion.risk_score:.3f}")

        feat = dict(lesion.features)
        raw_c = (feat.get("color_std_r", 0) + feat.get("color_std_g", 0) +
                 feat.get("color_std_b", 0)) / (3 * 100.0)
        feat["color_variance"] = float(np.clip(raw_c, 0.0, 1.0))

        for key, _label, max_val in self._ABCDE:
            value = feat.get(key, 0.0)
            trough, fill = self._bar_fills[key]
            trough.update_idletasks()
            tw = trough.winfo_width() or 140
            fill.place(x=0, y=0, relheight=1.0, width=int(tw * min(value / max_val, 1.0)))
            self._val_labels[key].config(text=f"{value:.3f}")

        segmenter = Segmenter(lesion.enhanced_image)
        segmenter._result = lesion.mask
        self._draw_on_canvas(segmenter.get_overlay())

        self._set_status(f"Analysis complete  ·  {lesion.risk_level} Risk  ({lesion.risk_score:.3f})")