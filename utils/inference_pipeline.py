import os
import time
import json
import csv
import cv2
import torch
import matplotlib
matplotlib.use("Agg")  # Prevent GUI backend error
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- Static setup ---
class_names = ['Caddisfly', 'Dipteran', 'Mayfly', 'Other', 'Stonefly', 'Terrestrial']
class_thresholds = {0: 0.4, 1: 0.4, 2: 0.4, 3: 0.0, 4: 0.4, 5: 0.4}
OTHER_IDX = 3
img_size = (224, 224)

# Load models once
model_cls = load_model('models/cls_model.keras')
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Inference will use device: {device}")
if device == "cuda":
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="models/detect_model.pt",
    confidence_threshold=0.2,
    device=device
)

def get_color_map(categories):
    base_colors = [
        (0, 0, 255),      # Red
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
    ]

    return {cat['id']: base_colors[i % len(base_colors)] for i, cat in enumerate(categories)}

def crop_predictions_from_sahi(image_path, sahi_result, conf_thresh=0.05):
    img = cv2.imread(image_path)
    crops = []
    annotations_map = {}
    for idx, obj in enumerate(sahi_result.object_prediction_list):
        if obj.score.value < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
        crop = img[y1:y2, x1:x2]
        crops.append(crop)
        filename = f"insect_{idx + 1}.jpg"
        annotations_map[filename] = {
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(obj.score.value),
            "category": None
        }
    return crops, annotations_map

def convert_to_coco(annotations_map, original_filename, image_path, image_id=1):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
    images = [{
        "id": image_id,
        "file_name": original_filename,
        "height": height,
        "width": width
    }]
    annotations = []
    ann_id = 1
    for fname, ann in annotations_map.items():
        if ann["category"] is None:
            continue
        category_id = class_names.index(ann["category"])
        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": ann["bbox"],
            "area": ann["bbox"][2] * ann["bbox"][3],
            "iscrowd": 0
        })
        ann_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}

def visualize_coco_annotations(coco_json_path, image_path, output_path):
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    annotations = coco['annotations']
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    color_map = get_color_map(coco['categories'])

    for ann in annotations:
        x, y, w, h = map(int, ann['bbox'])
        category_id = ann['category_id']
        color = color_map[category_id]
        label = categories[category_id]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"✅ Annotated image saved to: {output_path}")

def run_inference(image_path, output_dir, original_filename):
    print("\n🚀 Starting inference pipeline...")
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)
    crop_dir = os.path.join(output_dir, 'cropped_results')
    os.makedirs(crop_dir, exist_ok=True)

    detailed_csv = os.path.join(output_dir, 'detailed_predictions.csv')
    summary_csv = os.path.join(output_dir, 'class_summary.csv')
    coco_json = os.path.join(output_dir, 'coco_annotations.json')
    annotated_img_path = os.path.join(output_dir, 'annotated_output.jpg')

    # --- Detection ---
    print("🔍 Running SAHI detection...")
    org_img = cv2.imread(image_path)
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=752,
        slice_width=752,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3,
        postprocess_type="GREEDYNMM"
    )
    print(f"✅ Detected {len(result.object_prediction_list)} objects")

    # --- Cropping ---
    crops, annotations_map = crop_predictions_from_sahi(image_path, result)
    for idx, crop in enumerate(crops):
        cv2.imwrite(os.path.join(crop_dir, f"insect_{idx + 1}.jpg"), crop)
    print(f"✂️  Saved {len(crops)} crops")

    # --- Classification ---
    confidences = []
    class_counts = defaultdict(int)

    with open(detailed_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Raw Prediction', 'Confidence', 'Threshold', 'Final Prediction'])

        for file_name in sorted(os.listdir(crop_dir)):
            if not file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(crop_dir, file_name)
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            preds = model_cls.predict(img_batch, verbose=0)[0]
            pred_idx = np.argmax(preds)
            confidence = float(preds[pred_idx])
            confidences.append(confidence)
            threshold = class_thresholds.get(pred_idx, 0.5)

            final_idx = pred_idx if confidence >= threshold else OTHER_IDX
            final_class = class_names[final_idx]
            class_counts[final_class] += 1

            writer.writerow([
                file_name,
                class_names[pred_idx],
                round(confidence, 4),
                threshold,
                final_class
            ])

            if file_name in annotations_map:
                annotations_map[file_name]["category"] = final_class

        # Persist annotations_map so later edits can refer to all original detections
    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    avg_confidence = round(avg_confidence * 100, 2)
    ann_map_path = os.path.join(output_dir, 'annotations_map.json')
    try:
        with open(ann_map_path, 'w', encoding='utf-8') as af:
            json.dump(annotations_map, af, indent=2)
        print(f"✅ Wrote annotations_map to: {ann_map_path}")
    except Exception as e:
        print("Warning: failed to write annotations_map.json:", e)


    # --- Summary CSV ---
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Count'])
        for cls in class_names:
            writer.writerow([cls, class_counts[cls]])
            print(f"📊 {cls}: {class_counts[cls]}")

    # --- COCO + Annotated Image ---
    coco_dict = convert_to_coco(annotations_map, original_filename,image_path)
    with open(coco_json, 'w') as f:
        json.dump(coco_dict, f, indent=4)

    visualize_coco_annotations(coco_json, image_path, annotated_img_path)

    import zipfile

    # Define which folders to include in zip
    cropped_dir = os.path.join(output_dir, 'cropped_results')
    zip_path = os.path.join(output_dir, 'results.zip')

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(cropped_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, cropped_dir)  # zip relative to cropped_results/
                zipf.write(file_path, arcname)
                print(f"📦 Added to zip: {arcname}")


    print(f"📦 Class prediction folders zipped at: {zip_path}")

    print(f"✅ Finished in {time.time() - start_time:.2f} seconds")

    return {
        "class_counts": class_counts,
        "summary_csv": summary_csv,
        "detailed_csv": detailed_csv,
        "coco_json": coco_json,
        "annotated_img": annotated_img_path,
        "avg_confidence": avg_confidence,
    }



