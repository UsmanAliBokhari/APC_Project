"""
ui/theme.py
----------
A dark theme for our UI, the file includes 
the color pallette and fonts.
"""

import sys

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

# matching the fonts to the OS
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