"""
ui/app.py
----------
SkinScan – polished dark clinical theme.
Two-column workspace: images left, results right.
"""

from __future__ import annotations
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

from models.patient import Patient
from models.skin_lesion import SkinLesion
from models.scan_report import ScanReport
from processing.enhancer import Enhancer
from processing.segmenter import Segmenter
from processing.feature_extractor import FeatureExtractor
from detection.detector import Detector
from utils.helpers import (
    cv2_to_photoimage,
    load_image,
    risk_summary_text,
    filter_by_risk,
)


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
C_BG         = "#0d1117"
C_SURFACE    = "#161b22"
C_SURFACE_2  = "#1c2128"
C_SURFACE_3  = "#21262d"
C_SIDEBAR    = "#0d1117"

C_BORDER     = "#30363d"
C_BORDER_L   = "#3d444d"

C_TEXT       = "#e6edf3"
C_TEXT_2     = "#8b949e"
C_TEXT_3     = "#484f58"

C_ACCENT     = "#2f81f7"
C_ACCENT_H   = "#1f6feb"
C_ACCENT_DIM = "#1b3a6b"

C_LOW        = "#3fb950"
C_MED        = "#d29922"
C_HIGH       = "#f85149"

C_CANVAS_BG  = "#010409"

# compat aliases
C_PANEL      = C_SURFACE
C_HEADER     = "#010409"
C_HEADER_T   = C_TEXT
C_INPUT_BG   = C_SURFACE_2
C_INPUT_FG   = C_TEXT
C_INPUT_BD   = C_BORDER
C_ROW_ALT    = C_SURFACE_3
C_SIDEBAR_T  = C_TEXT
C_SIDEBAR_MUT = C_TEXT_2

RISK_COLOURS = {"Low": C_LOW, "Medium": C_MED, "High": C_HIGH}

if sys.platform == "darwin":
    F_UI    = "SF Pro Text"
    F_DISP  = "SF Pro Display"
    F_MONO  = "SF Mono"
else:
    F_UI    = "Helvetica"
    F_DISP  = "Helvetica"
    F_MONO  = "Courier"

FONT_TITLE  = (F_DISP, 14, "bold")
FONT_HEAD   = (F_UI,   11, "bold")
FONT_BODY   = (F_UI,   10)
FONT_SMALL  = (F_UI,    9)
FONT_MICRO  = (F_UI,    8)
FONT_MONO   = (F_MONO,  9)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _divider(parent, bg=C_BORDER, padx=0) -> tk.Frame:
    f = tk.Frame(parent, bg=bg, height=1)
    f.pack(fill="x", padx=padx)
    return f


def _btn(parent, text, bg, fg, cmd, state="normal", padx=14, pady=7, font=None):
    """Flat button with guaranteed foreground colour."""
    b = tk.Button(
        parent, text=text, command=cmd,
        bg=bg, fg=fg,
        font=font or FONT_SMALL,
        relief="flat", bd=0,
        cursor="hand2" if state == "normal" else "arrow",
        activebackground=bg, activeforeground=fg,
        disabledforeground=C_TEXT_3,
        padx=padx, pady=pady,
        state=state,
    )
    # Override: manually track enabled/disabled so fg stays correct
    _patch_btn_fg(b, fg)
    return b


def _patch_btn_fg(btn: tk.Button, enabled_fg: str):
    """Monkey-patch enable/disable so fg stays the right colour."""
    orig_config = btn.config

    def patched_config(**kw):
        if "state" in kw:
            if kw["state"] == "normal":
                kw.setdefault("fg", enabled_fg)
                kw.setdefault("cursor", "hand2")
            elif kw["state"] == "disabled":
                kw.setdefault("fg", C_TEXT_3)
                kw.setdefault("cursor", "arrow")
        return orig_config(**kw)

    btn.config  = patched_config
    btn.configure = patched_config


# ===========================================================================
# PatientSidebar
# ===========================================================================

class PatientSidebar(tk.Frame):
    def __init__(self, master, on_register, on_select, **kwargs):
        super().__init__(master, bg=C_SIDEBAR, **kwargs)
        self._on_register = on_register
        self._on_select   = on_select
        self._patients: list[Patient] = []
        self._patient_id_counter = 1
        self._build()

    def _build(self):
        # ── Logo ─────────────────────────────────────────────────────
        logo = tk.Frame(self, bg=C_SIDEBAR, padx=14)
        logo.pack(fill="x", pady=(18, 14))

        icon_box = tk.Frame(logo, bg=C_ACCENT_DIM, width=34, height=34)
        icon_box.pack(side="left")
        icon_box.pack_propagate(False)
        tk.Label(icon_box, text="🔬", font=(F_UI, 15),
                 bg=C_ACCENT_DIM, fg=C_TEXT).pack(expand=True)

        names = tk.Frame(logo, bg=C_SIDEBAR)
        names.pack(side="left", padx=(10, 0))
        tk.Label(names, text="SkinScan",
                 font=FONT_TITLE, bg=C_SIDEBAR, fg=C_TEXT, anchor="w").pack(anchor="w")
        tk.Label(names, text="Melanoma Detection",
                 font=FONT_MICRO, bg=C_SIDEBAR, fg=C_TEXT_2, anchor="w").pack(anchor="w")

        _divider(self, bg=C_BORDER)

        # ── Form ─────────────────────────────────────────────────────
        form = tk.Frame(self, bg=C_SIDEBAR, padx=14)
        form.pack(fill="x", pady=(12, 0))

        self.name_var = tk.StringVar()
        self.age_var  = tk.StringVar()
        self.loc_var  = tk.StringVar(value="Unknown")

        for lbl_text, attr in [("Name", "name_var"), ("Age", "age_var"), ("Body Location", "loc_var")]:
            tk.Label(form, text=lbl_text, font=FONT_MICRO,
                     bg=C_SIDEBAR, fg=C_TEXT_2, anchor="w").pack(fill="x", pady=(8, 2))

            wrap = tk.Frame(form, bg=C_BORDER, pady=1, padx=1)
            wrap.pack(fill="x")
            tk.Entry(wrap, textvariable=getattr(self, attr),
                     font=FONT_BODY, relief="flat", bd=0,
                     bg=C_INPUT_BG, fg=C_INPUT_FG,
                     insertbackground=C_ACCENT,
                     ).pack(fill="x", ipady=6, padx=1)

        tk.Label(form, text="Gender", font=FONT_MICRO,
                 bg=C_SIDEBAR, fg=C_TEXT_2, anchor="w").pack(fill="x", pady=(10, 4))

        self.gender_var = tk.StringVar(value="Unspecified")
        grp = tk.Frame(form, bg=C_SIDEBAR)
        grp.pack(fill="x")
        for val in ("Male", "Female", "Other"):
            tk.Radiobutton(grp, text=val, variable=self.gender_var, value=val,
                           bg=C_SIDEBAR, fg=C_TEXT,
                           selectcolor=C_ACCENT_DIM,
                           activebackground=C_SIDEBAR, activeforeground=C_TEXT,
                           font=FONT_SMALL).pack(side="left", padx=(0, 8))

        # Register btn
        reg = tk.Button(
            self, text="Register Patient",
            command=self._register,
            bg=C_ACCENT, fg="#ffffff",
            font=FONT_HEAD,
            relief="flat", bd=0, cursor="hand2",
            activebackground=C_ACCENT_H, activeforeground="#ffffff",
            pady=9,
        )
        reg.pack(fill="x", padx=14, pady=(14, 14))

        _divider(self, bg=C_BORDER)

        # ── History ───────────────────────────────────────────────────
        tk.Label(self, text="SESSION HISTORY", font=FONT_MICRO,
                 bg=C_SIDEBAR, fg=C_TEXT_3, anchor="w"
                 ).pack(anchor="w", padx=14, pady=(12, 6))

        lf = tk.Frame(self, bg=C_SIDEBAR)
        lf.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        sb = tk.Scrollbar(lf, orient="vertical", width=4,
                          bg=C_SIDEBAR, troughcolor=C_SIDEBAR,
                          relief="flat", bd=0)
        self._listbox = tk.Listbox(
            lf, yscrollcommand=sb.set,
            font=FONT_SMALL,
            bg=C_SIDEBAR, fg=C_TEXT,
            selectbackground=C_SURFACE_2, selectforeground=C_ACCENT,
            relief="flat", bd=0, activestyle="none", highlightthickness=0,
        )
        sb.config(command=self._listbox.yview)
        sb.pack(side="right", fill="y")
        self._listbox.pack(side="left", fill="both", expand=True)
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

    def _register(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Missing info", "Please enter the patient's name.")
            return
        try:
            age = int(self.age_var.get())
        except ValueError:
            messagebox.showwarning("Invalid age", "Age must be a whole number.")
            return
        patient = Patient(
            patient_id=f"PT-{self._patient_id_counter:04d}",
            name=name, age=age, gender=self.gender_var.get(),
            body_location=self.loc_var.get().strip() or "Unknown",
        )
        self._patient_id_counter += 1
        self._patients.append(patient)
        self._listbox.insert("end", f"  {patient.name}  ·  {patient.patient_id}")
        self._listbox.selection_clear(0, "end")
        self._listbox.selection_set("end")
        self._on_register(patient)

    def _on_listbox_select(self, _event):
        sel = self._listbox.curselection()
        if sel:
            self._on_select(self._patients[sel[0]])

    def refresh_entry(self, patient: Patient):
        try:
            idx = self._patients.index(patient)
        except ValueError:
            return
        risk = patient.overall_risk_level()
        colour = RISK_COLOURS.get(risk, C_TEXT_2)
        self._listbox.delete(idx)
        self._listbox.insert(idx, f"  {patient.name}  ·  {risk} Risk")
        self._listbox.itemconfig(idx, fg=colour)


# ===========================================================================
# ImagePanel  (left column of workspace)
# ===========================================================================

class ImagePanel(tk.Frame):
    """Displays two image canvases stacked, with workflow buttons above."""

    def __init__(self, master, on_load, on_enhance, on_segment, on_analyze, **kwargs):
        super().__init__(master, bg=C_SURFACE, **kwargs)
        self._on_load    = on_load
        self._on_enhance = on_enhance
        self._on_segment = on_segment
        self._on_analyze = on_analyze
        self._orig_photo = None
        self._proc_photo = None
        self._build()

    def _build(self):
        IMG_W, IMG_H = 420, 300

        # ── Top bar: title + buttons ──────────────────────────────────
        top = tk.Frame(self, bg=C_SURFACE, padx=14, pady=10)
        top.pack(fill="x")

        tk.Label(top, text="Image Processing",
                 font=FONT_HEAD, bg=C_SURFACE, fg=C_TEXT).pack(side="left")

        # Workflow buttons — right-aligned in title bar
        btn_row = tk.Frame(top, bg=C_SURFACE)
        btn_row.pack(side="right")

        self._btn_load = _btn(
            btn_row, "📂  Load", C_SURFACE_3, C_TEXT, self._on_load, padx=12)
        self._btn_enhance = _btn(
            btn_row, "✨  Enhance", C_SURFACE_3, C_TEXT, self._on_enhance,
            state="disabled", padx=12)
        self._btn_segment = _btn(
            btn_row, "⬡  Segment", C_SURFACE_3, C_TEXT, self._on_segment,
            state="disabled", padx=12)
        self._btn_analyze = _btn(
            btn_row, "▶  Analyse", C_ACCENT, C_TEXT, self._on_analyze,
            state="disabled", padx=14)

        for b in (self._btn_load, self._btn_enhance, self._btn_segment):
            b.pack(side="left", padx=(0, 6))
        self._btn_analyze.pack(side="left", padx=(8, 0))

        _divider(self, bg=C_BORDER)

        # ── Canvases ──────────────────────────────────────────────────
        canvas_area = tk.Frame(self, bg=C_SURFACE, padx=14, pady=14)
        canvas_area.pack(fill="both", expand=True)

        for col, label in enumerate(("Original", "Processed")):
            col_f = tk.Frame(canvas_area, bg=C_SURFACE)
            col_f.grid(row=0, column=col, padx=(0, 12 if col == 0 else 0), sticky="nsew")
            canvas_area.columnconfigure(col, weight=1)

            hdr = tk.Frame(col_f, bg=C_SURFACE)
            hdr.pack(fill="x", pady=(0, 6))
            dot_c = C_TEXT_3 if col == 0 else C_ACCENT
            tk.Label(hdr, text="●", font=(F_UI, 7),
                     bg=C_SURFACE, fg=dot_c).pack(side="left")
            tk.Label(hdr, text=f"  {label}", font=FONT_SMALL,
                     bg=C_SURFACE, fg=C_TEXT_2).pack(side="left")

            cv = tk.Canvas(col_f, width=IMG_W, height=IMG_H,
                           bg=C_CANVAS_BG,
                           highlightthickness=1,
                           highlightbackground=C_BORDER)
            cv.pack(fill="both", expand=True)
            cv.create_text(IMG_W // 2, IMG_H // 2,
                           text="No image" if col == 0 else "—",
                           fill=C_TEXT_3, font=FONT_SMALL, tags="placeholder")
            if col == 0:
                self._canvas_orig = cv
            else:
                self._canvas_proc = cv

        # Progress bar — shown during analysis
        self._progress = ttk.Progressbar(
            self, mode="indeterminate", length=300,
            style="Scan.Horizontal.TProgressbar")

    # ------------------------------------------------------------------

    def show_original(self, bgr_image):
        photo = cv2_to_photoimage(bgr_image, max_size=(420, 300))
        self._orig_photo = photo
        self._draw(self._canvas_orig, photo)

    def show_processed(self, bgr_image):
        photo = cv2_to_photoimage(bgr_image, max_size=(420, 300))
        self._proc_photo = photo
        self._draw(self._canvas_proc, photo)

    def enable_buttons(self, *names):
        m = {"enhance": self._btn_enhance,
             "segment": self._btn_segment,
             "analyze": self._btn_analyze}
        for n in names:
            if n in m:
                m[n].config(state="normal")

    def set_busy(self, busy):
        if busy:
            self._progress.pack(pady=(0, 10))
            self._progress.start(12)
        else:
            self._progress.stop()
            self._progress.pack_forget()

    @staticmethod
    def _draw(canvas, photo):
        canvas.delete("all")
        w = canvas.winfo_width() or 420
        h = canvas.winfo_height() or 300
        canvas.create_image(w // 2, h // 2, anchor="center", image=photo)


# ===========================================================================
# ResultsPanel  (right column of workspace, scrollable)
# ===========================================================================

class ResultsPanel(tk.Frame):
    _ABCDE_KEYS = [
        ("asymmetry",           "A  Asymmetry",           1.0),
        ("border_irregularity", "B  Border Irregularity",  1.0),
        ("color_variance",      "C  Colour Variance",      1.0),
        ("diameter_norm",       "D  Diameter",             1.0),
        ("texture_contrast",    "E  Texture Contrast",     2.0),
    ]

    def __init__(self, master, on_export, **kwargs):
        super().__init__(master, bg=C_SURFACE, **kwargs)
        self._on_export = on_export
        self._bar_fills: dict[str, tk.Frame]  = {}
        self._bar_troughs: dict[str, tk.Frame] = {}
        self._val_labels: dict[str, tk.Label] = {}
        self._build()

    def _build(self):
        # ── Top bar ───────────────────────────────────────────────────
        top = tk.Frame(self, bg=C_SURFACE, padx=14, pady=10)
        top.pack(fill="x")
        tk.Label(top, text="Analysis Results",
                 font=FONT_HEAD, bg=C_SURFACE, fg=C_TEXT).pack(side="left")
        self._export_btn = _btn(
            top, "Export Report", C_SURFACE_3, C_TEXT_2,
            self._on_export, state="disabled", padx=12, pady=6)
        self._export_btn.pack(side="right")

        _divider(self, bg=C_BORDER)

        # ── Scrollable canvas ─────────────────────────────────────────
        scroll_outer = tk.Frame(self, bg=C_SURFACE)
        scroll_outer.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(scroll_outer, bg=C_SURFACE,
                                 highlightthickness=0, bd=0)
        vsb = tk.Scrollbar(scroll_outer, orient="vertical",
                           command=self._canvas.yview,
                           width=6, relief="flat", bd=0,
                           bg=C_SURFACE_2, troughcolor=C_SURFACE)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        # Inner frame inside canvas
        self._inner = tk.Frame(self._canvas, bg=C_SURFACE)
        self._window = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw")

        self._inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        # Mouse wheel scrolling
        self._canvas.bind_all("<MouseWheel>",    self._on_mousewheel)
        self._canvas.bind_all("<Button-4>",      self._on_mousewheel)
        self._canvas.bind_all("<Button-5>",      self._on_mousewheel)

        self._build_inner(self._inner)

    def _on_inner_configure(self, _e):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self._canvas.itemconfig(self._window, width=e.width)

    def _on_mousewheel(self, e):
        if e.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif e.num == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def _build_inner(self, parent):
        pad = 16

        # ── Risk summary card ─────────────────────────────────────────
        card = tk.Frame(parent, bg=C_SURFACE_2,
                        highlightthickness=1, highlightbackground=C_BORDER)
        card.pack(fill="x", padx=pad, pady=(16, 12))

        inner = tk.Frame(card, bg=C_SURFACE_2, padx=16, pady=16)
        inner.pack(fill="x")

        # Coloured left stripe
        self._badge_stripe = tk.Frame(inner, bg=C_TEXT_3, width=4)
        self._badge_stripe.pack(side="left", fill="y")

        text_block = tk.Frame(inner, bg=C_SURFACE_2, padx=14)
        text_block.pack(side="left", fill="both", expand=True)

        self._risk_label = tk.Label(
            text_block, text="AWAITING ANALYSIS",
            font=(F_DISP, 16, "bold"), bg=C_SURFACE_2, fg=C_TEXT_2, anchor="w")
        self._risk_label.pack(anchor="w")

        self._score_lbl = tk.Label(
            text_block, text="Risk score  —",
            font=FONT_BODY, bg=C_SURFACE_2, fg=C_TEXT_3, anchor="w")
        self._score_lbl.pack(anchor="w", pady=(4, 0))

        # ── ABCDE section ─────────────────────────────────────────────
        tk.Label(parent, text="ABCDE CRITERIA", font=FONT_MICRO,
                 bg=C_SURFACE, fg=C_TEXT_3, anchor="w"
                 ).pack(anchor="w", padx=pad, pady=(4, 6))

        bars_wrap = tk.Frame(parent, bg=C_SURFACE)
        bars_wrap.pack(fill="x", padx=pad)

        # Column headers
        hdr = tk.Frame(bars_wrap, bg=C_SURFACE)
        hdr.pack(fill="x", pady=(0, 4))
        tk.Label(hdr, text="Criterion", width=22, anchor="w",
                 font=FONT_MICRO, bg=C_SURFACE, fg=C_TEXT_3).pack(side="left", padx=6)
        tk.Label(hdr, text="Score", anchor="w",
                 font=FONT_MICRO, bg=C_SURFACE, fg=C_TEXT_3).pack(side="left", padx=(8, 0))

        for i, (key, label, _) in enumerate(self._ABCDE_KEYS):
            row_bg = C_SURFACE_3 if i % 2 == 0 else C_SURFACE
            row = tk.Frame(bars_wrap, bg=row_bg)
            row.pack(fill="x")

            tk.Label(row, text=label, width=22, anchor="w",
                     font=FONT_SMALL, bg=row_bg, fg=C_TEXT,
                     padx=6, pady=7).pack(side="left")

            # Custom bar: trough + fill
            trough = tk.Frame(row, bg=C_SURFACE_2, height=6, width=180)
            trough.pack(side="left", padx=(6, 8))
            trough.pack_propagate(False)

            fill = tk.Frame(trough, bg=C_ACCENT, height=6, width=0)
            fill.place(x=0, y=0, relheight=1.0, width=0)

            self._bar_troughs[key] = trough
            self._bar_fills[key]   = fill

            val_lbl = tk.Label(row, text="—", width=7, anchor="w",
                               font=FONT_MONO, bg=row_bg, fg=C_TEXT)
            val_lbl.pack(side="left")
            self._val_labels[key] = val_lbl

        # ── Recommendations ───────────────────────────────────────────
        tk.Frame(parent, bg=C_BORDER, height=1).pack(fill="x", padx=pad, pady=(14, 0))

        tk.Label(parent, text="RECOMMENDATIONS", font=FONT_MICRO,
                 bg=C_SURFACE, fg=C_TEXT_3, anchor="w"
                 ).pack(anchor="w", padx=pad, pady=(12, 6))

        self._rec_text = tk.Text(
            parent, height=5, wrap="word",
            font=FONT_SMALL,
            bg=C_SURFACE_2, fg=C_TEXT,
            relief="flat", bd=0,
            padx=14, pady=12,
            state="disabled",
            highlightthickness=1,
            highlightbackground=C_BORDER,
            highlightcolor=C_ACCENT,
            insertbackground=C_ACCENT,
        )
        self._rec_text.pack(fill="x", padx=pad, pady=(0, 14))

        # Export btn inside scroll area too
        self._export_inner = _btn(
            parent, "💾  Export Report (.txt)",
            C_SURFACE_3, C_TEXT_2, self._on_export,
            state="disabled", padx=14, pady=8)
        self._export_inner.pack(anchor="w", padx=pad, pady=(0, 20))

    # ------------------------------------------------------------------

    def _set_bar(self, key, pct):
        trough = self._bar_troughs[key]
        fill   = self._bar_fills[key]
        trough.update_idletasks()
        tw = trough.winfo_width() or 180
        fill.place(x=0, y=0, relheight=1.0, width=int(tw * pct / 100))

    def update_report(self, report: ScanReport):
        colour = RISK_COLOURS.get(report.risk_level, C_TEXT_3)
        self._badge_stripe.config(bg=colour)
        self._risk_label.config(text=f"{report.risk_level.upper()} RISK", fg=colour)
        self._score_lbl.config(text=f"Risk score  {report.risk_score:.3f}")

        feat = dict(report.features)
        raw_c = (feat.get("color_std_r", 0) + feat.get("color_std_g", 0) +
                 feat.get("color_std_b", 0)) / (3 * 100.0)
        feat["color_variance"] = float(np.clip(raw_c, 0.0, 1.0))

        for key, _lbl, max_val in self._ABCDE_KEYS:
            value = feat.get(key, 0.0)
            self._set_bar(key, int(min(value / max_val, 1.0) * 100))
            self._val_labels[key].config(text=f"{value:.3f}")

        self._rec_text.config(state="normal")
        self._rec_text.delete("1.0", "end")
        for rec in report.recommendations:
            self._rec_text.insert("end", f"·  {rec}\n")
        self._rec_text.config(state="disabled")

        self._export_btn.config(state="normal")
        self._export_inner.config(state="normal")

        # Scroll to top after update
        self._canvas.yview_moveto(0)

    def clear(self):
        self._badge_stripe.config(bg=C_TEXT_3)
        self._risk_label.config(text="AWAITING ANALYSIS", fg=C_TEXT_2)
        self._score_lbl.config(text="Risk score  —")
        for key in self._bar_fills:
            self._set_bar(key, 0)
            self._val_labels[key].config(text="—")
        self._rec_text.config(state="normal")
        self._rec_text.delete("1.0", "end")
        self._rec_text.config(state="disabled")
        self._export_btn.config(state="disabled")
        self._export_inner.config(state="disabled")


# ===========================================================================
# SkinScanApp
# ===========================================================================

class SkinScanApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._current_patient: Optional[Patient] = None
        self._current_lesion:  Optional[SkinLesion]  = None
        self._current_report:  Optional[ScanReport]  = None
        self._detector = Detector()
        self._configure_window()
        self._configure_styles()
        self._build_ui()

    def _configure_window(self):
        self.root.title("SkinScan – Melanoma Detection System")
        self.root.geometry("1300x860")
        self.root.minsize(1000, 680)
        self.root.configure(bg=C_BG)

    def _configure_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TFrame",         background=C_BG)
        s.configure("Sidebar.TFrame", background=C_SIDEBAR)
        s.configure("Scan.Horizontal.TProgressbar",
                    troughcolor=C_SURFACE_3, background=C_ACCENT, thickness=3)

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=C_HEADER, height=48)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="SkinScan  ·  Melanoma Early Detection",
                 font=FONT_HEAD, bg=C_HEADER, fg=C_TEXT
                 ).pack(side="left", padx=18, pady=12)

        self._ts_label = tk.Label(header, text="", font=FONT_SMALL,
                                  bg=C_HEADER, fg=C_TEXT_3)
        self._ts_label.pack(side="right", padx=18)
        self._update_clock()

        tk.Frame(self.root, bg=C_BORDER, height=1).pack(fill="x", side="top")

        # ── Body ──────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill="both", expand=True)

        # Sidebar
        self._sidebar = PatientSidebar(
            body,
            on_register=self._on_patient_registered,
            on_select=self._on_patient_selected,
        )
        self._sidebar.pack(side="left", fill="y")
        tk.Frame(body, bg=C_BORDER, width=1).pack(side="left", fill="y")

        # Workspace — two-column grid
        workspace = tk.Frame(body, bg=C_BG)
        workspace.pack(side="left", fill="both", expand=True)
        workspace.columnconfigure(0, weight=3)  # images get more width
        workspace.columnconfigure(1, weight=2)  # results panel
        workspace.rowconfigure(0, weight=1)

        # Left col: image panel
        self._image_panel = ImagePanel(
            workspace,
            on_load=self._on_load_image,
            on_enhance=self._on_enhance,
            on_segment=self._on_segment,
            on_analyze=self._on_analyze,
        )
        self._image_panel.grid(row=0, column=0, sticky="nsew",
                               padx=(12, 6), pady=12)

        # Vertical divider
        tk.Frame(workspace, bg=C_BORDER, width=1).grid(
            row=0, column=1, sticky="ns", pady=12)

        # Right col: results panel
        self._results_panel = ResultsPanel(
            workspace,
            on_export=self._on_export_report,
        )
        self._results_panel.grid(row=0, column=1, sticky="nsew",
                                 padx=(7, 12), pady=12)

        # ── Status bar ────────────────────────────────────────────────
        tk.Frame(self.root, bg=C_BORDER, height=1).pack(fill="x", side="bottom")
        sbar = tk.Frame(self.root, bg=C_HEADER, height=26)
        sbar.pack(fill="x", side="bottom")
        sbar.pack_propagate(False)

        self._status_dot = tk.Label(sbar, text="●", font=(F_UI, 7),
                                    bg=C_HEADER, fg=C_TEXT_3)
        self._status_dot.pack(side="left", padx=(12, 4), pady=5)

        self._status_var = tk.StringVar(value="Ready  ·  Register a patient to begin.")
        tk.Label(sbar, textvariable=self._status_var,
                 font=FONT_SMALL, bg=C_HEADER, fg=C_TEXT_2, anchor="w"
                 ).pack(side="left", pady=5)

    # ------------------------------------------------------------------

    def _update_clock(self):
        self._ts_label.config(text=datetime.now().strftime("%a %d %b %Y  %H:%M:%S"))
        self.root.after(1000, self._update_clock)

    def _set_status(self, msg, dot=None):
        self._status_var.set(msg)
        self._status_dot.config(fg=dot or C_TEXT_3)

    # ------------------------------------------------------------------
    # Patient callbacks
    # ------------------------------------------------------------------

    def _on_patient_registered(self, patient):
        self._current_patient = patient
        self._current_lesion  = None
        self._current_report  = None
        self._results_panel.clear()
        self._image_panel.enable_buttons()
        self._set_status(
            f"Registered: {patient.name}  [{patient.patient_id}]  ·  Load an image to begin.",
            C_ACCENT)

    def _on_patient_selected(self, patient):
        self._current_patient = patient
        self._set_status(f"Switched to: {patient.name}  [{patient.patient_id}]")

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _on_load_image(self):
        if self._current_patient is None:
            messagebox.showwarning("No patient",
                "Please register or select a patient before loading an image.")
            return
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
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        lesion = SkinLesion(image_path=path,
                            body_location=self._current_patient.body_location)
        lesion.raw_image = img
        self._current_patient.add_lesion(lesion)
        self._current_lesion = lesion
        self._image_panel.show_original(img)
        self._image_panel.enable_buttons("enhance", "segment", "analyze")
        self._results_panel.clear()
        self._set_status(f"Loaded: {lesion.image_name}  ({w}×{h} px)", C_ACCENT)

    # ------------------------------------------------------------------
    # Processing callbacks
    # ------------------------------------------------------------------

    def _on_enhance(self):
        if not self._check_lesion():
            return
        try:
            enhancer = Enhancer(self._current_lesion.raw_image, method="pipeline")
            self._current_lesion.enhanced_image = enhancer.result
            self._image_panel.show_processed(self._current_lesion.enhanced_image)
            self._set_status("Enhancement complete  (CLAHE → denoise → sharpen).", C_LOW)
        except Exception as exc:
            messagebox.showerror("Enhancement error", str(exc))

    def _on_segment(self):
        if not self._check_lesion():
            return
        try:
            segmenter = Segmenter(self._current_lesion.working_image, method="otsu")
            self._current_lesion.mask = segmenter.result
            self._image_panel.show_processed(segmenter.get_overlay())
            px = int(np.sum(self._current_lesion.mask > 0))
            self._set_status(
                f"Segmentation complete  ·  Lesion area: {px:,} px.", C_LOW)
        except Exception as exc:
            messagebox.showerror("Segmentation error", str(exc))

    def _on_analyze(self):
        if not self._check_lesion():
            return
        if self._current_patient is None:
            messagebox.showwarning("No patient", "Register a patient first.")
            return
        self._image_panel.set_busy(True)
        self._set_status("Running analysis…", C_MED)
        threading.Thread(target=self._analysis_worker, daemon=True).start()

    def _analysis_worker(self):
        try:
            lesion = self._current_lesion
            if lesion.enhanced_image is None:
                lesion.enhanced_image = Enhancer(lesion.raw_image).result
            if lesion.mask is None:
                lesion.mask = Segmenter(lesion.enhanced_image).result
            extractor = FeatureExtractor(lesion.enhanced_image, lesion.mask)
            lesion.features = extractor.result
            risk_score, risk_level = self._detector.predict(lesion.features)
            lesion.risk_score = risk_score
            lesion.risk_level = risk_level
            report = ScanReport(patient=self._current_patient, lesion=lesion,
                                risk_score=risk_score, risk_level=risk_level,
                                features=lesion.features)
            lesion.report = report
            self._current_report = report
            self.root.after(0, self._on_analysis_done, report, None)
        except Exception as exc:
            self.root.after(0, self._on_analysis_done, None, exc)

    def _on_analysis_done(self, report, error):
        self._image_panel.set_busy(False)
        if error:
            messagebox.showerror("Analysis error", str(error))
            self._set_status("Analysis failed — see error dialog.", C_HIGH)
            return
        self._results_panel.update_report(report)
        segmenter = Segmenter(report.lesion.enhanced_image)
        segmenter._result = report.lesion.mask
        self._image_panel.show_processed(segmenter.get_overlay())
        self._sidebar.refresh_entry(self._current_patient)
        risk_dot = RISK_COLOURS.get(report.risk_level, C_TEXT_2)
        self._set_status(
            f"Analysis complete  ·  {self._current_patient.name}  ·  "
            f"{report.risk_level} Risk  ({report.risk_score:.3f})",
            risk_dot)

    def _on_export_report(self):
        if self._current_report is None:
            messagebox.showinfo("Nothing to export", "Run an analysis first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save report", defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=(
                f"report_{self._current_patient.patient_id}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"))
        if not path:
            return
        saved = self._current_report.export_txt(path)
        messagebox.showinfo("Report saved", f"Report written to:\n{saved}")
        self._set_status(f"Report exported  →  {saved}", C_LOW)

    def _check_lesion(self):
        if self._current_lesion is None or self._current_lesion.raw_image is None:
            messagebox.showwarning("No image", "Please load a lesion image first.")
            return False
        return True