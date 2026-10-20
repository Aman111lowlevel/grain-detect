# 🌾 Automated Rice Grain Quality Detection & CVAT Annotation Pipeline

An automated computer vision and image processing pipeline that segments individual rice grains from high-resolution flatbed scanner images, performs multi-label physical quality classification, and packages the results directly into a CVAT-compatible **YOLO 1.1** annotation archive.

> **For Juniors / Team Members:**  
> This script does all the heavy lifting for you! It automatically detects every individual rice grain, analyzes its physical quality attributes (Whole, Broken, Chalky, Damaged, Discoloured, Dehusked, Foreign Matter), and generates `rice_annotations.zip`.  
> **You do not need to draw bounding boxes manually in CVAT** — simply run this script and import the generated zip into CVAT!

---

## 🚀 Quick Start Guide (Instructions for Juniors)

### 1. Requirements & Setup

Make sure you have Python 3.8+ installed along with `opencv-python` and `numpy`:

```bash
pip install opencv-python numpy
```

---

### 2. Where to Change Dataset Paths in Code

Open [`grain_detect.py`](grain_detect.py) and check **Line 23**:

```python
# ==========================================
# Configuration and Constants
# ==========================================
DEFAULT_INPUT_DIR = r"C:\Users\AMAN\Downloads\rice_samples"  # <--- CHANGE THIS PATH
```

Change `DEFAULT_INPUT_DIR` to the folder on your computer where your scanner images are stored.

#### 💡 Or Pass Path via Command Line (No code changes needed!)
You can also run the script with the folder path directly in your terminal:

```bash
python grain_detect.py "C:\path\to\your\rice_samples"
```

---

### 3. Run the Pipeline

Execute the script from the project folder:

```bash
python grain_detect.py
```

The script will:
1. Scan your dataset folder for valid scanner images (`Scan77.jpg`, `Scan78.jpg`, etc.).
2. Automatically ignore non-scan files, timestamp downloads (e.g. `1791737568681.jpg`), and temporary crops.
3. Enhance contrast, segment touching grains using Distance Transform + Watershed, and detect multi-label defects.
4. Output a summary table with counts per image and per quality class.
5. Create **`rice_annotations.zip`** in the project folder.

---

### 4. Uploading Directly into CVAT

Once `rice_annotations.zip` is created:

1. Open your **CVAT** task or job (make sure the task contains the corresponding `Scan*.jpg` images).
2. Click the top-left **Actions** menu $\rightarrow$ **Upload Annotations**.
3. Select format: **`YOLO 1.1`**.
4. Drag and drop or browse to select **`rice_annotations.zip`**.
5. Click **Submit**.

🎉 **Your annotations are done!** CVAT will automatically load all normalized bounding boxes with their exact multi-label classes.

---

## 🏷️ Rice Quality Classes (Exact CVAT Order)

The pipeline evaluates every detected grain against the 7 standardized CVAT classes:

| Class ID | Class Label | Description & Detection Logic |
| :---: | :--- | :--- |
| **0** | **Whole** | Full-length standard unbroken rice grains. Flagged when aspect ratio $\ge 1.85$ and not broken or foreign debris. |
| **1** | **Chalky** | Grains with an opaque, chalk-white central core (due to incomplete starch filling). Detected by high inner core brightness ($V > 185$) with uniform low variance ($\sigma < 18$). |
| **2** | **Broken** | Kernels broken during milling. Flagged when aspect ratio $< 1.85$ AND contour area $< 65\%$ of the image median grain area. |
| **3** | **Damaged** | Insect pecks, fungal spots, or necrosis. Flagged when localized dark spots ($\ge 10$ pixels) are detected inside the grain core. |
| **4** | **Discoloured** | Heat-damaged, fermented, or yellowish/brown grains. Flagged when mean Saturation (HSV) $> 35$ or LAB $b^*$ channel $> 138$. |
| **5** | **Dehusked** | Grains with rough brownish husk texture, elevated Laplacian texture variance, or reddish/tan pericarp streaks. |
| **6** | **Foreign Matter** | Stones, chaff, plastic, dust, or non-grain debris. Flagged if solidity $< 0.68$, extreme aspect ratios ($> 5.5$ or $< 1.05$), or dark non-grain objects. |

---

## 🔬 Multi-Label Tagging Logic

Real rice grains often possess multiple defect attributes at the same time:
- A grain can be both **Broken** and **Chalky**.
- A grain can be **Whole** and **Damaged** (e.g. an unbroken grain with an insect peck).
- A grain can be **Whole** and **Discoloured**.

### How YOLO 1.1 Handles Multi-Label Annotations
For multi-attribute grains, the pipeline emits multiple lines in the annotation file with the **exact same normalized coordinates**:

```text
# Example: Grain at coordinates (0.391, 0.913) tagged as both Whole (0) and Damaged (3)
0 0.391373 0.913815 0.010588 0.019247
3 0.391373 0.913815 0.010588 0.019247
```

CVAT natively reads these co-located bounding boxes as multiple labels assigned to that object.

---

## 🛠️ High-Recall Computer Vision Pipeline

```
Raw Scanner Image (ScanXX.jpg)
       │
       ▼
Grayscale Conversion + CLAHE Contrast Enhancement
       │
       ▼
Dual Thresholding (Otsu + Intensity-Gated Adaptive Gaussian)
       │
       ▼
Morphological Opening (Clean Sensor Noise)
       │
       ▼
Contour Area Filter (Retain fragments > 15 px)
       │
       ▼
Touching Cluster Detection (Area > 7,500 px)
       │
       ├─► [Cluster] ──► Distance Transform (cv2.DIST_L2)
       │                       │
       │                       ▼
       │                 Local Peak Finding (sure_fg)
       │                       │
       │                       ▼
       │                 Watershed Segmentation ──► Split Individual Grains
       │
       └─► [Single Grain]
               │
               ▼
Multi-Label Attribute Evaluation (Geometry, HSV, LAB, Core vs Edge, Texture)
               │
               ▼
YOLO Coordinate Normalization (x_center, y_center, width, height)
               │
               ▼
Packaging into CVAT YOLO 1.1 Archive ('rice_annotations.zip')
```

---

## 📦 Output Archive Structure

`rice_annotations.zip` is structured strictly according to CVAT YOLO 1.1 requirements:

```text
rice_annotations.zip
├── obj.names                  # Exact list of the 7 quality classes
├── obj.data                   # Darknet/CVAT config (classes = 7, train = train.txt)
├── train.txt                  # Image paths index (prevents 'subset list' import errors)
└── obj_train_data/
    ├── Scan77.txt             # Normalized bounding boxes for Scan77.jpg
    ├── Scan78.txt             # Normalized bounding boxes for Scan78.jpg
    ├── Scan80.txt
    ├── scan79.txt
    └── ...
```

---

## 📊 Benchmark Statistics (15 Scanner Images)

- **Total Grains Detected:** `4,890` (increased recall by $+687$ grains via watershed cluster splitting)
- **Single-Label Grains:** `3,399` ($69.5\%$)
- **Multi-Label Grains:** `1,491` ($30.5\%$)
- **Total YOLO Bounding Box Entries:** `6,579`
- **Average Runtime:** $\approx 1.8$ seconds per 35-Megapixel ($7014 \times 5100$ px) image.

---

## 👨‍💻 Maintainer

Developed by [Aman](https://github.com/Aman111lowlevel)  
Repository: [https://github.com/Aman111lowlevel/grain-detect](https://github.com/Aman111lowlevel/grain-detect)
