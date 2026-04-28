"""
Main.py
--------
Entry point for SkinScan.

Usage:
    python Main.py

What happens on startup:
    1.  utils/model_setup.ensure_model_exists() checks whether
        models/detector.pkl already exists.  If not, it generates a
        synthetic training dataset and trains a RandomForestClassifier,
        then serialises it.  This takes ~5 seconds on the first run only.
    2.  A Tkinter root window is created and SkinScanApp is launched.
"""

import tkinter as tk
from utils.model_setup import ensure_model_exists
from ui.app import SkinScanApp


def main() -> None:
    # Ensure a trained model exists before the window opens
    ensure_model_exists()

    root = tk.Tk()
    app = SkinScanApp(root)    # noqa: F841 – keeps the reference alive

    # Centre the window on screen
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    root.mainloop()


if __name__ == "__main__":
    main()
