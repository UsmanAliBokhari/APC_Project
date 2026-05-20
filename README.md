# SkinScan — Melanoma Early Detection

A desktop application for dermoscopic image analysis using the ABCDE rule for melanoma screening. Built with Python, Tkinter, and a manual image processing pipeline.

---

## Features

- Load and display dermoscopy images
- Automatic image enhancement (CLAHE + edge-aware sharpening)
- Lesion segmentation with Otsu thresholding and morphological cleanup
- ABCDE feature extraction (Asymmetry, Border, Colour, Diameter, Texture)
- Risk classification — Low / Medium / High
- Segmentation overlay visualisation

---

## Project Structure

```
APC_Project/
├── Main.py                        # Entry point
├── requirements.txt
│
├── ui/
│   ├── skin_scan_app.py           # Main application window
│   ├── theme.py                   # Colours, fonts, design tokens
│   └── widgets.py                 # Reusable UI helpers
│
├── models/
│   ├── skin_lesion.py             # Lesion domain object
│
├── processing/
│   ├── image_processor.py         # Abstract base class
│   ├── enhancer.py                # CLAHE + sharpening pipeline
│   ├── segmenter.py               # Lesion mask extraction
│   └── feature_extractor.py      # ABCDE feature computation
│
├── detection/
│   └── detector.py                # Risk classification model
│
├── utils/
│   ├── helpers.py                 # load_image, img_to_photoimage, etc.
│   └── model_setup.py             # Model training and serialisation
│
└── test_images/                   # Sample dermoscopy images for testing
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourname/APC_Project.git
cd APC_Project
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the application**
```bash
python Main.py
```

---

## Requirements


```
numpy==2.4.6
opencv-python==4.13.0.92
Pillow==12.2.0
scikit-image==0.26.0
scikit-learn==1.8.0
scipy==1.17.1
joblib==1.5.3
```


---

## How It Works

### Pipeline

```
Load Image → Enhance → Segment → Extract Features → Classify → Display
```

**Enhance** — CLAHE in LAB colour space boosts lesion contrast without shifting hue. Edge-aware unsharp masking sharpens boundaries using a Laplacian edge map so flat skin regions are left untouched.

**Segment** — Otsu thresholding on the grayscale image with circle cropping to exclude dermoscope vignetting. Morphological open/close cleans noise, and the largest connected component is kept as the lesion ROI.

**Extract** — Five ABCDE features are computed inside the segmented mask:
- **A** Asymmetry — bitwise XOR of flipped masks
- **B** Border — compactness index from contour area/perimeter
- **C** Colour — per-channel mean and std inside the mask
- **D** Diameter — bounding box size normalised to image diagonal
- **E** Texture — GLCM contrast via scikit-image

**Classify** — A trained `RandomForestClassifier` maps the feature vector to a risk score and level.

---

## Usage

1. Launch the app with `python Main.py`
2. Click **Load** to open a dermoscopy image — sample images are included in `test_images/`, or download more from the [ISIC Archive](https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery)
3. Click **Analyse** to run the full pipeline
4. The segmentation overlay and ABCDE scores appear automatically

---

## Credits

Test images sourced from the [ISIC Archive](https://www.isic-archive.com),
contributed by the ViDIR Group, Department of Dermatology, Medical University of Vienna.
Licensed under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/).