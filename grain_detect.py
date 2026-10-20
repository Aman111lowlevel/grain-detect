"""
Upgraded High-Recall Rice Grain Detection & Multi-Label Classification Pipeline.

Features:
1. High-Recall Preprocessing:
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) to amplify faint and shadowed grains.
   - Dual thresholding: Combined Otsu thresholding + Gated Adaptive Gaussian thresholding.
   - Distance Transform + Watershed segmentation to cleanly split touching grain clusters.
   - Comprehensive contour filtering (15 px to 35,000 px) to retain small broken fragments and split grains.

2. Multi-Label Classification:
   - Evaluates each grain for multiple concurrent attributes:
     * Whole (0): Elongated standard grain (mutually exclusive with Broken and Foreign Matter).
     * Chalky (1): High-intensity opaque whiteness with low variance across the inner core.
     * Broken (2): Aspect ratio < 1.85 and area < (0.65 * median_area).
     * Damaged (3): Localized dark/black necrotic spots within the grain core.
     * Discoloured (4): Elevated Saturation (HSV > 35) or yellow/brown LAB b* deviation.
     * Dehusked (5): Rough brownish husk texture, high gradient variance, or pericarp layer.
     * Foreign Matter (6): Irregular non-grain shape, low solidity (< 0.68), or extreme aspect ratio.

3. Multi-Label YOLO 1.1 Export:
   - Emits multiple bounding box entries with identical normalized coordinates for multi-attribute grains.
   - Generates 'obj.names', 'obj.data', and 'train.txt' strictly formatted for CVAT.
   - Packages into 'rice_annotations.zip'.
"""

import os
import sys
import re
import shutil
import zipfile
from collections import Counter, OrderedDict
import cv2
import numpy as np


# ==========================================
# Configuration and Constants
# ==========================================
DEFAULT_INPUT_DIR = r"C:\Users\AMAN\Downloads\rice_samples"
OUTPUT_DIR = "obj_train_data"
OUTPUT_ZIP = "rice_annotations.zip"
OBJ_NAMES_FILE = "obj.names"
OBJ_DATA_FILE = "obj.data"
TRAIN_TXT_FILE = "train.txt"

# CVAT Labels (Exact order)
LABELS = [
    "Whole",            # 0
    "Chalky",           # 1
    "Broken",           # 2
    "Damaged",          # 3
    "Discoloured",      # 4
    "Dehusked",         # 5
    "Foreign Matter"    # 6
]

# Morphological kernels
KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_ERODE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))


def is_valid_scan_image(filename):
    """
    Validates scan image files. Excludes timestamped downloads (e.g., 1791737568681.jpg),
    previews, crops, and temporary files.
    """
    lower = filename.lower()
    if not lower.endswith((".jpg", ".jpeg", ".png")):
        return False
    if lower.startswith(("preview_", "crop_", "temp_", "test_")):
        return False
    name_without_ext = os.path.splitext(filename)[0]
    if name_without_ext.isdigit():
        return False
    if re.match(r"^(scan|img_scan|rice_scan)[\-_]?\d+$", name_without_ext, re.IGNORECASE):
        return True
    if "scan" in lower:
        return True
    return False


def segment_grains_high_recall(img, gray):
    """
    Segments grains using CLAHE, combined Otsu + Adaptive thresholding,
    and Distance Transform + Watershed cluster splitting.
    
    Returns:
        isolated_grains (list of contours): All detected and separated grains.
        split_cluster_count (int): Number of clusters split into individual grains.
    """
    h, w = gray.shape

    # 1. CLAHE contrast enhancement to amplify faint/shadowed grains
    enhanced = CLAHE.apply(gray)

    # 2. Dual Thresholding: Otsu + Gated Adaptive Gaussian Thresholding
    ret, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, -20)
    
    # Gate adaptive threshold by intensity floor to avoid flat scanner background noise
    floor_mask = np.uint8(gray > (ret * 0.65)) * 255
    gated_adapt = cv2.bitwise_and(adapt, floor_mask)
    combined = cv2.bitwise_or(otsu, gated_adapt)

    # 3. Suppress scanner plate outer margins
    margin = 25
    clean_mask = combined.copy()
    clean_mask[:margin, :] = 0
    clean_mask[-margin:, :] = 0
    clean_mask[:, :margin] = 0
    clean_mask[:, -margin:] = 0

    # 4. Morphological opening to clean sensor noise
    opening = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, KERNEL_OPEN)

    # 5. Extract initial contours
    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    isolated_grains = []
    split_cluster_count = 0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 15 or area > 35000:
            continue  # Dust (<15 px) or plate border (>35,000 px)

        # Single grains: area <= 7500 px
        if area <= 7500:
            isolated_grains.append(c)
        else:
            # Touching grain cluster: split using Distance Transform + Watershed
            bx, by, bw, bh = cv2.boundingRect(c)
            roi_mask = np.zeros((bh, bw), dtype=np.uint8)
            shifted_c = c - np.array([bx, by])
            cv2.drawContours(roi_mask, [shifted_c], -1, 255, -1)

            # Local distance transform
            dist_roi = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_roi, 0.40 * dist_roi.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)

            num_labels, markers = cv2.connectedComponents(sure_fg)
            if num_labels > 2:
                # 2 or more distinct grain centers in cluster
                sure_bg = cv2.dilate(roi_mask, KERNEL_OPEN, iterations=2)
                unknown = cv2.subtract(sure_bg, sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0

                roi_bgr = img[by:by+bh, bx:bx+bw]
                cv2.watershed(roi_bgr, markers)

                grains_added_from_cluster = 0
                for m_id in range(2, num_labels + 1):
                    sub_mask = np.uint8(markers == m_id)
                    sub_cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for sc in sub_cnts:
                        if cv2.contourArea(sc) > 15:
                            isolated_grains.append(sc + np.array([bx, by]))
                            grains_added_from_cluster += 1
                
                if grains_added_from_cluster > 1:
                    split_cluster_count += 1
            else:
                if area < 12000:
                    isolated_grains.append(c)

    return isolated_grains, split_cluster_count


def classify_grain_multilabel(cnt, img, gray, med_area):
    """
    Evaluates every isolated grain contour for concurrent multi-label attributes.
    Returns:
        labels (list of int): List of active class IDs (0-6).
    """
    area = cv2.contourArea(cnt)
    bx, by, bw, bh = cv2.boundingRect(cnt)

    # 1. Geometry & Aspect Ratio
    rect = cv2.minAreaRect(cnt)
    (rx, ry), (rw, rh), _ = rect
    length = max(rw, rh)
    width = max(min(rw, rh), 1.0)
    aspect_ratio = length / width

    hull = cv2.convexHull(cnt)
    solidity = area / max(cv2.contourArea(hull), 1.0)

    # 2. Extract ROI and Masks
    mask = np.zeros((bh, bw), dtype=np.uint8)
    shifted = cnt - np.array([bx, by])
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    roi_bgr = img[by:by+bh, bx:bx+bw]
    roi_gray = gray[by:by+bh, bx:bx+bw]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

    grain_hsv = roi_hsv[mask == 255]
    grain_lab = roi_lab[mask == 255]

    mean_sat = float(np.mean(grain_hsv[:, 1]))
    mean_val = float(np.mean(grain_hsv[:, 2]))
    mean_b = float(np.mean(grain_lab[:, 2]))   # Yellow/brown tint
    mean_a = float(np.mean(grain_lab[:, 1]))   # Red/brown tint

    # 3. Core vs Edge Analysis
    core_mask = cv2.erode(mask, KERNEL_ERODE, iterations=2)
    has_core = np.sum(core_mask == 255) > 15

    if has_core:
        core_vals = roi_hsv[core_mask == 255, 2]
        core_mean = float(np.mean(core_vals))
        core_std = float(np.std(core_vals))
        edge_mask = cv2.subtract(mask, core_mask)
        edge_mean = float(np.mean(roi_hsv[edge_mask == 255, 2])) if np.sum(edge_mask == 255) > 0 else core_mean
        necrotic_spots = int(np.sum(core_vals < (core_mean * 0.78)))
    else:
        core_mean = mean_val
        core_std = float(np.std(grain_hsv[:, 2]))
        edge_mean = mean_val
        necrotic_spots = 0

    # Texture variance (for husk/roughness detection)
    lap = cv2.Laplacian(roi_gray, cv2.CV_64F)
    lap_var = float(np.var(lap[mask == 255]))

    # 4. Multi-Label Evaluation
    labels = []

    # Flag 6: Foreign Matter (irregular debris, non-grain geometry)
    is_foreign = (solidity < 0.68 or aspect_ratio > 5.5 or aspect_ratio < 1.05 or mean_val < 50.0)
    if is_foreign:
        labels.append(6)

    # Flag 2: Broken (low aspect ratio & small area relative to dataset median)
    is_broken = (aspect_ratio < 1.85 and area < (0.65 * med_area))
    if is_broken:
        labels.append(2)

    # Flag 0: Whole (full-length grain; mutually exclusive with Broken and Foreign)
    if not is_broken and not is_foreign and aspect_ratio >= 1.85:
        labels.append(0)

    # Flag 1: Chalky (opaque high-intensity core whiteness with uniform low variance)
    if core_mean > 185.0 and core_std < 18.0 and (core_mean >= edge_mean):
        labels.append(1)

    # Flag 4: Discoloured (high saturation or yellow/brown tinting)
    if mean_sat > 35.0 or mean_b > 138.0:
        labels.append(4)

    # Flag 3: Damaged (localized dark/necrotic spots inside core)
    if necrotic_spots >= 10:
        labels.append(3)

    # Flag 5: Dehusked (rough brownish husk texture, high gradient variance, or pericarp layer)
    if (25.0 < mean_sat <= 35.0 and mean_b > 134.0) or (mean_a > 131.0) or (lap_var > 650.0 and mean_sat > 25.0):
        labels.append(5)

    # Default fallback
    if not labels:
        labels.append(0)

    return sorted(labels)


def process_single_image(img_path, output_dir):
    """
    Performs high-recall segmentation and multi-label YOLO annotation export.
    Returns:
        txt_path (str)
        num_grains (int)
        multi_label_count (int)
        class_counts (Counter)
        total_annotations (int)
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [ERROR] Could not read image: {img_path}")
        return None, 0, 0, Counter(), 0

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. High-Recall Segmentation
    grains, split_clusters = segment_grains_high_recall(img, gray)

    # Calculate median area across detected grains
    areas = [cv2.contourArea(c) for c in grains]
    med_area = float(np.median(areas)) if areas else 0.0

    # 2. Multi-Label Tagging & YOLO Generation
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    txt_filename = f"{base_name}.txt"
    txt_path = os.path.join(output_dir, txt_filename)

    class_counts = Counter()
    multi_label_grains = 0
    annotations = []

    for cnt in grains:
        active_labels = classify_grain_multilabel(cnt, img, gray, med_area)

        if len(active_labels) > 1:
            multi_label_grains += 1

        bx, by, bw, bh = cv2.boundingRect(cnt)
        x_center = (bx + bw / 2.0) / float(w)
        y_center = (by + bh / 2.0) / float(h)
        norm_w = float(bw) / float(w)
        norm_h = float(bh) / float(h)

        # Emit an entry for each active class with identical coordinates
        for cid in active_labels:
            class_counts[cid] += 1
            annotations.append(f"{cid} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    # Write YOLO annotation file
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in annotations:
            f.write(line + "\n")

    return txt_path, len(grains), multi_label_grains, class_counts, len(annotations)


def create_metadata(processed_txt_files):
    """
    Creates obj.names, obj.data, and train.txt strictly formatted for CVAT YOLO 1.1.
    """
    # 1. obj.names with exact 7 labels
    with open(OBJ_NAMES_FILE, "w", encoding="utf-8") as f:
        for label in LABELS:
            f.write(label + "\n")

    # 2. obj.data pointing directly to train = train.txt
    with open(OBJ_DATA_FILE, "w", encoding="utf-8") as f:
        f.write(f"classes = {len(LABELS)}\n")
        f.write(f"train = {TRAIN_TXT_FILE}\n")
        f.write(f"names = {OBJ_NAMES_FILE}\n")
        f.write("backup = backup/\n")

    # 3. train.txt listing relative paths to every image/label
    with open(TRAIN_TXT_FILE, "w", encoding="utf-8") as f:
        for txt_file in sorted(processed_txt_files):
            base_name = os.path.basename(txt_file)
            f.write(f"data/obj_train_data/{base_name}\n")


def build_zip_archive(zip_path, txt_files):
    """
    Packages EXACTLY:
      1. obj.names
      2. obj.data
      3. train.txt
      4. obj_train_data/<ScanXX>.txt files
    into the ZIP archive.
    """
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(OBJ_NAMES_FILE, OBJ_NAMES_FILE)
        zipf.write(OBJ_DATA_FILE, OBJ_DATA_FILE)
        zipf.write(TRAIN_TXT_FILE, TRAIN_TXT_FILE)

        for txt_file in sorted(txt_files):
            arc_name = os.path.join(OUTPUT_DIR, os.path.basename(txt_file)).replace("\\", "/")
            zipf.write(txt_file, arc_name)

    # Verification
    with zipfile.ZipFile(zip_path, "r") as test_zip:
        bad_file = test_zip.testzip()
        if bad_file:
            raise RuntimeError(f"Corrupt file in archive: {bad_file}")
        namelist = test_zip.namelist()

    return namelist


def main():
    print("=" * 90)
    print("   UPGRADED HIGH-RECALL & MULTI-LABEL CVAT YOLO 1.1 RICE GRAIN PIPELINE   ")
    print("=" * 90)

    input_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_DIR
    if not os.path.exists(input_dir):
        print(f"[ERROR] Directory '{input_dir}' not found.")
        sys.exit(1)

    print(f"Dataset Directory: {input_dir}")

    # Gather ONLY valid scan images
    all_files = os.listdir(input_dir)
    scan_files = sorted([f for f in all_files if is_valid_scan_image(f)])

    print(f"Found {len(scan_files)} valid scan image(s):")
    for f in scan_files:
        print(f"  - {f}")
    print()

    if not scan_files:
        print("[ERROR] No valid scan images found.")
        sys.exit(1)

    # 1. Clean old output directory and archives
    for item in [OUTPUT_DIR, OUTPUT_ZIP, OBJ_NAMES_FILE, OBJ_DATA_FILE, TRAIN_TXT_FILE]:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Process each image with high-recall segmentation and multi-label tagging
    results = OrderedDict()
    total_class_counts = Counter()
    processed_txt_files = []
    total_grains_detected = 0
    total_multilabel_grains = 0
    total_yolo_annotations = 0

    for idx, filename in enumerate(scan_files, 1):
        img_path = os.path.join(input_dir, filename)
        print(f"[{idx:02d}/{len(scan_files):02d}] Processing: {filename}...", end="", flush=True)

        txt_path, n_grains, n_multi, counts, n_annot = process_single_image(img_path, OUTPUT_DIR)
        if txt_path is not None:
            results[filename] = {
                "grains": n_grains,
                "multi": n_multi,
                "counts": counts,
                "annotations": n_annot
            }
            total_class_counts.update(counts)
            total_grains_detected += n_grains
            total_multilabel_grains += n_multi
            total_yolo_annotations += n_annot
            processed_txt_files.append(txt_path)
            print(f" done ({n_grains} grains, {n_multi} multi-label, {n_annot} boxes)")
        else:
            print(" FAILED")

    # 3. Create metadata files (obj.names, obj.data, train.txt)
    create_metadata(processed_txt_files)

    # 4. Package clean zip archive with all 4 elements
    archive_contents = build_zip_archive(OUTPUT_ZIP, processed_txt_files)

    # 5. Display Summary Table
    print("\n" + "=" * 105)
    print("                                   BATCH EXECUTION SUMMARY                                   ")
    print("=" * 105)
    header = f"{'Image Filename':<16} | {'Grains':<6} | {'Multi':<5} | {'Boxes':<5} | " + " | ".join(f"{lbl[:7]:<7}" for lbl in LABELS)
    print(header)
    print("-" * len(header))

    for filename, data in results.items():
        counts = data["counts"]
        cols = [f"{counts[cid]:<7}" for cid in range(len(LABELS))]
        print(f"{filename:<16} | {data['grains']:<6} | {data['multi']:<5} | {data['annotations']:<5} | " + " | ".join(cols))

    print("-" * len(header))
    total_cols = [f"{total_class_counts[cid]:<7}" for cid in range(len(LABELS))]
    print(f"{'TOTALS':<16} | {total_grains_detected:<6} | {total_multilabel_grains:<5} | {total_yolo_annotations:<5} | " + " | ".join(total_cols))
    print("=" * 105)

    # Multi-label vs Single-label breakdown
    single_label_count = total_grains_detected - total_multilabel_grains
    single_pct = (single_label_count / total_grains_detected * 100.0) if total_grains_detected > 0 else 0.0
    multi_pct = (total_multilabel_grains / total_grains_detected * 100.0) if total_grains_detected > 0 else 0.0

    print("\nMulti-Label Tagging Statistics:")
    print(f"  Total Isolated Grains       : {total_grains_detected:,}")
    print(f"  Single-Label Grains         : {single_label_count:,} ({single_pct:.1f}%)")
    print(f"  Multi-Label Grains          : {total_multilabel_grains:,} ({multi_pct:.1f}%)")
    print(f"  Total YOLO Bounding Boxes   : {total_yolo_annotations:,}")

    # Class Breakdown Percentages
    print("\nClass Tag Distribution Across All Grains:")
    for cid, label in enumerate(LABELS):
        count = total_class_counts[cid]
        pct = (count / total_yolo_annotations * 100.0) if total_yolo_annotations > 0 else 0.0
        print(f"  [{cid}] {label:<15}: {count:5d} tags ({pct:5.1f}%)")

    # Archive Structure Verification
    zip_size_kb = os.path.getsize(OUTPUT_ZIP) / 1024.0
    print("\n" + "=" * 105)
    print("                       VERIFIED CVAT YOLO 1.1 ARCHIVE CONTENTS                       ")
    print("=" * 105)
    print(f"Archive: {os.path.abspath(OUTPUT_ZIP)} ({zip_size_kb:.1f} KB)")
    print(f"Total Entries: {len(archive_contents)}")
    for item in archive_contents:
        print(f"  [+] {item}")
    print("=" * 105)


if __name__ == "__main__":
    main()