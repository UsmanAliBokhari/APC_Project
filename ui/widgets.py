"""
ui/widgets.py
----------
Small reusable UI helpers 
"""
from ui.theme import *
import tkinter as tk

def _divider(parent, bg=C_BORDER, padx=0) -> tk.Frame:
    """Create a 1px horizontal divider line."""
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
