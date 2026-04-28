# SkinScan – Code Notes
## Melanoma Detection App · APC Project 2025–2026

---

## Quick Start

```bash
pip install -r requirements.txt
python Main.py
```

The first run trains the model (~5 s). Subsequent runs start instantly.

---

## Project Structure

```
SkinScan/
├── Main.py                         ← single entry point
├── requirements.txt
│
├── models/                         ← domain objects (pure data + logic)
│   ├── patient.py
│   ├── skin_lesion.py
│   └── scan_report.py
│
├── processing/                     ← image processing (OOP hierarchy)
│   ├── image_processor.py          ← abstract base class
│   ├── enhancer.py                 ← Enhancer(ImageProcessor)
│   ├── segmenter.py                ← Segmenter(ImageProcessor)
│   └── feature_extractor.py       ← FeatureExtractor(ImageProcessor)
│
├── detection/
│   └── detector.py                 ← loads model, classifies lesions
│
├── ui/
│   └── app.py                      ← full Tkinter GUI (4 classes)
│
└── utils/
    ├── helpers.py                  ← map/filter/reduce utilities
    └── model_setup.py              ← auto-trains RandomForest on first run
```

---

## File-by-File Reference

### `Main.py`

Entry point. Does two things:

1. Calls `utils.model_setup.ensure_model_exists()` — if `models/detector.pkl`
   is missing it trains and saves the classifier. This is the only time
   training happens.
2. Creates the `tk.Tk` root window, instantiates `SkinScanApp`, centres
   the window, and calls `mainloop()`.

---

### `models/patient.py` — `Patient`

A **dataclass** holding demographic info and a list of `SkinLesion` objects.

| Method | Purpose |
|---|---|
| `add_lesion(lesion)` | Appends lesion and back-links `lesion.patient_id` |
| `overall_risk_level()` | Returns highest risk across all lesions (High > Medium > Low > Pending) |
| `analysed_lesions()` | Filters to only fully-processed lesions |

The `lesions` list is the primary example of **multiple instantiation** of
one class type (many `SkinLesion` objects per `Patient`).

---

### `models/skin_lesion.py` — `SkinLesion`

Holds every processing artefact for one image in a single object:

| Attribute | Type | Set by |
|---|---|---|
| `raw_image` | `np.ndarray` (BGR uint8) | `SkinScanApp._on_load_image` |
| `enhanced_image` | `np.ndarray` | `Enhancer` |
| `mask` | `np.ndarray` (binary 0/255) | `Segmenter` |
| `features` | `dict[str, float]` | `FeatureExtractor` |
| `risk_score` | `float` in [0, 1] | `Detector` |
| `risk_level` | `"Low"/"Medium"/"High"` | `Detector` |
| `report` | `ScanReport` | `SkinScanApp._analysis_worker` |

The `working_image` property returns `enhanced_image` if available,
otherwise falls back to `raw_image` — this is used by `Segmenter`
when enhancement was skipped.

---

### `models/scan_report.py` — `ScanReport`

Assembled after classification. Holds references to the patient and lesion,
plus auto-generated recommendations keyed by risk level.

`export_txt(path)` writes a human-readable plaintext report and returns
the written path (used by the export button in the GUI).

---

### `processing/image_processor.py` — `ImageProcessor` (abstract base)

All three processing classes inherit from this.

```
ImageProcessor (ABC)
    ├── Enhancer
    ├── Segmenter
    └── FeatureExtractor
```

The contract: each subclass receives an image in `__init__`, calls
`super().__init__(image)`, which immediately calls `self._process()`.
The subclass must set `self._result` inside `_process`. The `result`
property then exposes it.

Shared utility static methods (`to_gray`, `to_float`, `to_uint8`) live
here and are available to all subclasses — a textbook use of inheritance.

**Why inheritance here?** All three processors share the same lifecycle
(receive image → process → expose result) and the same utility methods.
The abstract `_process` method enforces the contract without repeating
boilerplate in each subclass.

---

### `processing/enhancer.py` — `Enhancer(ImageProcessor)`

Improves image quality before segmentation. Supports four `method` values:

| Method | What it does |
|---|---|
| `"clahe"` | CLAHE in LAB colour space — boosts contrast without hue shift |
| `"denoise"` | Non-local means — removes hair/speckle noise |
| `"sharpen"` | Unsharp mask — accentuates lesion edges |
| `"pipeline"` | All three in sequence (default, best for dermoscopy) |

**CLAHE** operates on the L (luminance) channel of the LAB colour space
rather than BGR directly, so it enhances contrast without changing
the colour information used by the ABCDE C criterion.

---

### `processing/segmenter.py` — `Segmenter(ImageProcessor)`

Isolates the lesion ROI as a binary mask. Three strategies:

| Method | When to use |
|---|---|
| `"otsu"` | Default. Fast, robust for most dermoscopy images |
| `"kmeans"` | Better for multicolour lesions — clusters by BGR |
| `"adaptive"` | Handles uneven lighting / shadows |

After thresholding, `_postprocess` runs:
1. **Morphological closing** — fills small gaps in the mask
2. **Morphological opening** — removes speckle noise
3. **Largest connected component** — discards background artefacts
4. **Flood-fill hole-filling** — fills any internal holes in the lesion

`get_overlay()` produces a display image: the original with a
cyan-yellow tint over the mask and a contour drawn on the boundary.

---

### `processing/feature_extractor.py` — `FeatureExtractor(ImageProcessor)`

Computes ABCDE features. `result` returns a `dict[str, float]`.

| Feature key(s) | ABCDE | Algorithm |
|---|---|---|
| `asymmetry` | A | Flip mask, measure XOR overlap vs total area |
| `border_irregularity` | B | 1 − (4π·Area / Perimeter²) |
| `color_mean_b/g/r`, `color_std_b/g/r` | C | Per-channel stats within mask |
| `diameter_norm` | D | MinAreaRect major axis / image diagonal |
| `texture_contrast/homogeneity/energy/correlation` | E | GLCM via scikit-image |

The GLCM is computed at distances [1, 2] and four angles, then averaged —
this captures texture in all orientations.

**Note:** `result` is overridden to return the `dict` instead of the raw
numpy array (the base class `_result` array is still populated for
internal use by the `detector`).

---

### `detection/detector.py` — `Detector`

Loads `models/detector.pkl` (a scikit-learn `Pipeline`) and exposes:

```python
risk_score, risk_level = detector.predict(feature_dict)
```

Internally:
- The feature dict is converted to a `(1, 13)` float32 array in a
  fixed column order (`_FEATURE_ORDER`) that must match training.
- The pipeline applies `StandardScaler` then `RandomForestClassifier`.
- `predict_proba` gives probability of malignancy (class 1).
- Thresholds: `< 0.35` → Low, `0.35–0.65` → Medium, `≥ 0.65` → High.

The model is loaded **lazily** on first call to avoid startup delay.

---

### `utils/model_setup.py`

Generates a synthetic training dataset and trains the classifier. Only
runs when `models/detector.pkl` is absent.

**Synthetic data**: 1200 benign + 800 malignant samples. Each feature is
drawn from a Gaussian whose mean/std is parameterised by class, informed
by published ABCDE rule statistics (e.g. malignant lesions have higher
asymmetry, border irregularity, colour std, and texture contrast).

The trained object is a scikit-learn `Pipeline`:
```
StandardScaler → RandomForestClassifier(n_estimators=200, max_depth=8)
```

---

### `utils/helpers.py`

Shared utilities. This file is the primary showcase of:

| Python feature | Function | How |
|---|---|---|
| `filter()` | `risk_summary_text`, `filter_by_risk` | Select analysed/matching lesions |
| `map()` | `risk_summary_text`, `average_risk_score` | Extract risk level / score fields |
| `reduce()` | `risk_summary_text`, `average_risk_score` | Accumulate counts / sum scores |
| list comprehension | `risk_summary_text` | Build "Low ×1, High ×2" display strings |

`cv2_to_photoimage` handles the BGR→RGB→PIL→PhotoImage conversion pipeline
needed to display OpenCV arrays in Tkinter canvases.

---

### `ui/app.py` — Four UI classes

Follows a **passive-view pattern**: panels hold no domain state; all logic
lives in `SkinScanApp`.

#### `PatientSidebar(ttk.Frame)`
Patient registration form (Name, Age, Body Location, Gender) + scrollable
session history listbox. Fires `on_register` and `on_select` callbacks.
After analysis, `refresh_entry(patient)` colours the patient entry by risk.

#### `ImagePanel(ttk.LabelFrame)`
Two dark canvases side by side (Original / Processed) plus four action
buttons: Load, Enhance, Segment, Run Analysis. Buttons are disabled until
the prerequisite step is done. An indeterminate `ttk.Progressbar` spins
during background analysis.

#### `ResultsPanel(ttk.LabelFrame)`
- Risk badge (coloured label: green/amber/red)
- Five `ttk.Progressbar` bars, one per ABCDE criterion
- Recommendations text box
- Export button (writes `.txt` via `ScanReport.export_txt`)

#### `SkinScanApp`
Root controller. Key design decisions:

- **Background thread for analysis** (`threading.Thread` + `root.after`):
  the three heavy steps (enhance, segment, classify) run in a daemon
  thread. Results are posted back to the main thread via `root.after(0, …)`
  — the only Tkinter-safe way to update widgets from a non-main thread.
- **Lazy processing**: Enhance and Segment can be run interactively before
  clicking Run Analysis. If skipped, the analysis worker runs them
  automatically so the button always works.

---

## OOP Grading Criteria Checklist

| Criterion | Where |
|---|---|
| `map` | `utils/helpers.py` → `risk_summary_text`, `average_risk_score` |
| `filter` | `utils/helpers.py` → `risk_summary_text`, `filter_by_risk` |
| `reduce` | `utils/helpers.py` → `risk_summary_text`, `average_risk_score` |
| List comprehension | `utils/helpers.py` → `risk_summary_text`, `batch_risk_levels` |
| Multiple instantiation | `Patient.lesions` holds many `SkinLesion` objects per session |
| Inheritance | `Enhancer`, `Segmenter`, `FeatureExtractor` all subclass `ImageProcessor` |
| Single command | `python Main.py` |
| OOP structure | `Patient` owns `SkinLesion`; `Detector` produces `ScanReport` |

---

## How the Processing Pipeline Works (end-to-end)

```
Raw image (BGR)
    │
    ▼  Enhancer.pipeline
CLAHE → Denoise → Sharpen
    │
    ▼  Segmenter.otsu
Otsu threshold → Morphological cleanup → Largest blob → Hole fill
    │
    ▼  FeatureExtractor
Asymmetry · Border · Colour · Diameter · Texture (GLCM)
    │
    ▼  Detector.predict
StandardScaler → RandomForest → probability → risk level
    │
    ▼  ScanReport
Recommendations + export to .txt
```

---

## Notes on the ML Model

The model is trained on **synthetic data** — real production use would
require training on the [ISIC Archive](https://www.isic-archive.com/)
dataset. The synthetic data is parameterised from published ABCDE
statistics so the feature distributions are medically plausible, but the
model should not be used for actual clinical decisions.

**Disclaimer (also in the exported report):** SkinScan is a screening
aid only. It does not replace clinical diagnosis by a qualified
dermatologist.
