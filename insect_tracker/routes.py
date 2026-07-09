import os
import re
import secrets
import shutil
import json
import csv
import cv2
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, send_from_directory, jsonify
from insect_tracker import app, db, bcrypt
from insect_tracker.forms import UploadTrapImage
from utils.inference_pipeline import run_inference  # uses output_dir you pass
from wtforms.validators import DataRequired
import json
from roboflow import Roboflow
from dotenv import load_dotenv
import uuid

# -------------------------------
# Load environment variables from .env
# -------------------------------
load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

THRESHOLD_FILE = "threshold.json"
STEP = 50
START = 100


# ------------------------------
# Helpers for file saving
# ------------------------------
def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static', 'profile_pics', picture_fn)
    os.makedirs(os.path.dirname(picture_path), exist_ok=True)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)
    return picture_fn

def save_trap_picture(form_picture, request_id):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext

    folder = os.path.join(app.root_path, 'static', 'trap_images', request_id)
    os.makedirs(folder, exist_ok=True)

    picture_path = os.path.join(folder, picture_fn)

    i = Image.open(form_picture)
    i = i.convert("RGB")
    i.save(picture_path, quality=100)

    return picture_fn, folder

# ------------------------------
# Basic pages
# ------------------------------
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html', title='About')

# ------------------------------
# Upload route (single output directory)
# ------------------------------
@app.route("/upload", methods=['GET', 'POST'])
def upload():
    form = UploadTrapImage()
    if form.validate_on_submit():
        request_id = str(uuid.uuid4())
        if form.picture.data:
           
            picture_file, trap_folder = save_trap_picture(form.picture.data, request_id)
            image_path = os.path.join(trap_folder, picture_file)
            metadata = {
                "sample_id": form.sample_id.data or "",
                "side_or_trapnum": form.side_or_trapnum.data or "",
                "watershed": form.watershed.data or "",
                "date": form.date.data.strftime('%m/%d/%Y') if form.date.data else ""
            }

            # Base output folder
            output_root = os.path.join(app.root_path, 'static', 'output')
            os.makedirs(output_root, exist_ok=True)

            # Unique folder per request
            output_dir = os.path.join(output_root, request_id)
            os.makedirs(output_dir, exist_ok=True)

            # Ensure the subfolders expected by inference exist (inference will create them too)
            os.makedirs(os.path.join(output_dir, 'cropped_results'), exist_ok=True)

            # Run inference pipeline (it writes outputs into output_dir) ===
            original_filename = form.picture.data.filename
            results = run_inference(image_path=image_path, output_dir=output_dir, original_filename=original_filename, metadata=metadata)

            # === 6. Pass results to the template ===
            # NOTE: we point urls to static/output/...
            return render_template(
                'upload.html',
                title='Image Upload',
                form=form,
                processed=True,
                class_counts=results.get('class_counts', {}),
                avg_confidence=results.get('avg_confidence', 0),
                request_id=request_id,
                annotated_img=url_for('static', filename=f'output/{request_id}/annotated_output.jpg'),
                summary_csv=url_for('download_file', filename=f'output/{request_id}/class_summary.csv'),
                detailed_csv=url_for('download_file', filename=f'output/{request_id}/detailed_predictions.csv'),
                zip_path=url_for('download_file', filename=f'output/{request_id}/results.zip'),
                coco_json=url_for('download_file', filename=f'output/{request_id}/coco_annotations.json'),
            )

    return render_template('upload.html', title='Image Upload', form=form)

# ------------------------------
# Download helper
# ------------------------------
@app.route('/download/<path:filename>')
def download_file(filename):
    # filename is relative to static/; e.g. "output/class_summary.csv"
    return send_from_directory(os.path.join(app.root_path, 'static'), filename, as_attachment=True)

# ------------------------------
# Utility paths (single output)
# ------------------------------
def _paths(request_id):
    base = os.path.join(app.root_path, 'static', 'output', request_id)
    return {
        "output_dir": base,
        "cropped_dir": os.path.join(base, "cropped_results"),
        "detailed_csv": os.path.join(base, "detailed_predictions.csv"),
        "summary_csv": os.path.join(base, "class_summary.csv"),
        "coco_json": os.path.join(base, "coco_annotations.json"),
        "annotations_map": os.path.join(base, "annotations_map.json"),
        "annotated_img": os.path.join(base, "annotated_output.jpg"),
    }

def _read_detailed_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def _write_detailed_csv(path, rows):
    fieldnames = ['Image Name', 'Raw Prediction', 'Confidence', 'Threshold', 'Final Prediction']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(out)

def _find_trap_image_path(request_id):
    trap_dir = os.path.join(app.root_path, 'static', 'trap_images', request_id)
    if not os.path.isdir(trap_dir):
        return None
    imgs = [f for f in os.listdir(trap_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not imgs:
        return None
    return os.path.join(trap_dir, imgs[0])


# Keep class_names consistent with inference_pipeline.py
class_names = ['Caddisfly', 'Dipteran', 'Mayfly', 'Other', 'Stonefly', 'Terrestrial']
NOT_INSECT_LABEL = 'Not an Insect'

# ------------------------------
# crop_list: return crops (only those marked Other by default)
# ------------------------------
@app.route('/crop_list')
def crop_list():
    """
    JSON: { items: [ { filename, raw_prediction, confidence, final, url }, ... ] }
    This function uses static/output (single folder).
    It prefers detailed_predictions.csv and returns only rows whose Final Prediction == 'Other'.
    Falls back to listing cropped_results folder if CSV missing.
    """
    request_id = request.args.get("request_id")
    if not request_id:
        return jsonify({"error": "missing request_id"}), 400

    p = _paths(request_id)
    base = p['output_dir']
    cropped = p['cropped_dir']
    edited_detailed_csv = os.path.join(base, 'detailed_predictions_edited.csv')
    detailed_csv = edited_detailed_csv if os.path.exists(edited_detailed_csv) else p['detailed_csv']

    items = []
    app.logger.debug("crop_list: looking in %s", base)

    if os.path.exists(detailed_csv):
        try:
            with open(detailed_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    fname = (r.get('Image Name') or '').strip()
                    if not fname:
                        continue
                    
                        # build url relative to static/
                    rel = os.path.join('output', request_id, 'cropped_results', fname).replace('\\', '/')
                    url = url_for('static', filename=rel)
                    items.append({
                        "filename": fname,
                        "raw_prediction": r.get('Raw Prediction', ''),
                        "confidence": r.get('Confidence', ''),
                        "final": r.get('Final Prediction', '') or 'Other',
                        "edited_to": r.get('Edited_to', 'N/A'),
                        "url": url
                    })
        except Exception as e:
            app.logger.warning("crop_list: failed to read detailed CSV %s: %s", detailed_csv, e)

    return jsonify({"items": items})


@app.route('/update_crop', methods=['POST'])
def update_crop():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "no json body"}), 400

    request_id = data.get("request_id")
    if not request_id:
        return jsonify({"error": "missing request_id"}), 400

    filename = data.get('filename')
    new_class = data.get('new_class') or "Other"

    p = _paths(request_id)
    detailed_csv = p['detailed_csv']
    output_dir = p['output_dir']
    summary_csv = p['summary_csv']

    edited_detailed_csv = os.path.join(output_dir, 'detailed_predictions_edited.csv')
    edited_summary_csv = os.path.join(output_dir, 'class_summary_edited.csv')

    # KEY FIX: load edited CSV if exists, otherwise original
    if os.path.exists(edited_detailed_csv):
        rows = _read_detailed_csv(edited_detailed_csv)
    else:
        rows = _read_detailed_csv(detailed_csv)

    if not rows:
        return jsonify({"error": "CSV not found or empty"}), 400

    found = False

    for r in rows:
        if r.get('Image Name') == filename:
            r['Edited_to'] = new_class
            found = True
        else:
            # preserve previous edits
            r['Edited_to'] = r.get('Edited_to', 'N/A')

    if not found:
        return jsonify({"error": "filename not found"}), 404

    # --- Write edited detailed CSV ---
    fieldnames = [
        'Image Name',
        'Raw Prediction',
        'Confidence',
        'Threshold',
        'Final Prediction',
        'Edited_to'
    ]

    try:
        with open(edited_detailed_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        return jsonify({"error": "failed writing edited CSV", "detail": str(e)}), 500

    # --- Recompute summary from edited rows ---
    edited_counts = {c: 0 for c in class_names}

    for r in rows:
        final = r.get('Edited_to')
        if not final or final == 'N/A':
            final = r.get('Final Prediction')

        edited_counts[final] = edited_counts.get(final, 0) + 1

    metadata = {}
    with open(summary_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        first_row = next(reader)

        metadata["sample_id"] = first_row.get("sample_id", "")
        metadata["side_or_trapnum"] = first_row.get("side_or_trapnum", "")
        metadata["watershed"] = first_row.get("watershed", "")
        metadata["date"] = first_row.get("date", "")

    try:
        with open(edited_summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id','side_or_trapnum','watershed', 'date','dipteran_small', 'terrestrial_small', 'caddisfly_large', 'stonefly_large', 'mayfly_large', 'other_small'])
            writer.writerow([metadata["sample_id"], metadata["side_or_trapnum"], metadata["watershed"], metadata["date"], edited_counts.get('Dipteran', 0), edited_counts.get('Terrestrial', 0), edited_counts.get('Caddisfly', 0), edited_counts.get('Stonefly', 0), edited_counts.get('Mayfly', 0), edited_counts.get('Other', 0)])
    except Exception as e:
        return jsonify({"error": "failed writing summary", "detail": str(e)}), 500
    
    # --- Rebuild annotations using edited CSV ---
    coco_json_edited = os.path.join(output_dir, 'coco_annotations_edited.json')
    annotations_map_edited = os.path.join(output_dir, 'annotations_map_edited.json')

    # Load from the edited map if present (preserves manually-added annotations),
    # otherwise fall back to the original map.
    ann_map_source = annotations_map_edited if os.path.exists(annotations_map_edited) else p['annotations_map']
    if os.path.exists(ann_map_source):
        try:
            with open(ann_map_source, 'r', encoding='utf-8') as f:
                ann_map_edited = json.load(f)
        except Exception as e:
            app.logger.warning(f"Failed to load {ann_map_source}: {e}")
            ann_map_edited = {}
    else:
        ann_map_edited = {}

    for r in rows:
        fname = r.get('Image Name')
        edited = r.get('Edited_to')

        if fname in ann_map_edited:
            if edited and edited != 'N/A':
                if edited == NOT_INSECT_LABEL:
                    ann_map_edited[fname]['category'] = None
                else:
                    ann_map_edited[fname]['category'] = edited

    # Save edited annotation map
    try:
        with open(annotations_map_edited, 'w', encoding='utf-8') as f:
            json.dump(ann_map_edited, f, indent=2)
    except Exception as e:
        app.logger.warning(f"Failed to write annotations_map_edited.json: {e}")

    # --- Generate COCO from edited annotations ---
    try:
        from utils.inference_pipeline import convert_to_coco
    except Exception as e:
        app.logger.warning(f"Could not import convert_to_coco: {e}")
        convert_to_coco = None

    # Get original image
    trap_out_dir = os.path.join(app.root_path, 'static', 'trap_images', request_id)
    imgs = [f for f in os.listdir(trap_out_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    orig_image_path = os.path.join(trap_out_dir, imgs[0]) if imgs else None

    # Get original filename from original COCO
    coco_json_path = p['coco_json']
    original_fname = None

    if os.path.exists(coco_json_path):
        try:
            with open(coco_json_path, 'r', encoding='utf-8') as f:
                old = json.load(f)
            if old.get('images'):
                original_fname = old['images'][0].get('file_name')
        except Exception:
            pass

    if convert_to_coco and orig_image_path:
        try:
            coco_dict = convert_to_coco(
                ann_map_edited,
                original_filename=original_fname,
                image_path=orig_image_path
            )

            with open(coco_json_edited, 'w', encoding='utf-8') as f:
                json.dump(coco_dict, f, indent=2)
        except Exception as e:
            app.logger.warning(f"Failed to generate edited COCO JSON: {e}")
    # --- Return files ---
    edited_files_rel = {
        "summary_csv_edited": f"output/{request_id}/class_summary_edited.csv",
        "detailed_csv_edited": f"output/{request_id}/detailed_predictions_edited.csv",
        "coco_json_edited": f"output/{request_id}/coco_annotations_edited.json"
    }

    return jsonify({"ok": True, "counts": edited_counts, "edited_files": edited_files_rel})


@app.route('/trap_image')
def trap_image():
    """Serve the original full-resolution trap image (dynamically resolved,
    since the file may have been renamed by a prior Roboflow upload)."""
    request_id = request.args.get("request_id")
    if not request_id:
        return jsonify({"error": "missing request_id"}), 400

    path = _find_trap_image_path(request_id)
    if not path:
        return jsonify({"error": "trap image not found"}), 404

    directory, filename = os.path.split(path)
    return send_from_directory(directory, filename)


@app.route('/annotation_overlay')
def annotation_overlay():
    """JSON: { boxes: [ { filename, bbox, category }, ... ] } for currently
    active (non-removed) annotations, used to draw reference boxes."""
    request_id = request.args.get("request_id")
    if not request_id:
        return jsonify({"error": "missing request_id"}), 400

    p = _paths(request_id)
    edited_map_path = os.path.join(p['output_dir'], 'annotations_map_edited.json')
    map_path = edited_map_path if os.path.exists(edited_map_path) else p['annotations_map']

    ann_map = {}
    if os.path.exists(map_path):
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                ann_map = json.load(f)
        except Exception as e:
            app.logger.warning(f"annotation_overlay: failed to read {map_path}: {e}")

    boxes = [
        {"filename": fname, "bbox": ann.get("bbox"), "category": ann.get("category")}
        for fname, ann in ann_map.items()
        if ann.get("category")
    ]
    return jsonify({"boxes": boxes})


@app.route('/add_annotation', methods=['POST'])
def add_annotation():
    """Add a brand-new, manually-drawn annotation. Reuses the same
    '_edited' sidecar files as /update_crop, so a manual box becomes
    indistinguishable from an edited one once saved."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "no json body"}), 400

    request_id = data.get("request_id")
    if not request_id:
        return jsonify({"error": "missing request_id"}), 400

    bbox = data.get("bbox")
    class_name = data.get("class_name")

    if not bbox or len(bbox) != 4:
        return jsonify({"error": "invalid bbox"}), 400
    if class_name not in class_names:
        return jsonify({"error": "invalid class_name"}), 400

    try:
        x, y, w, h = (int(round(float(v))) for v in bbox)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid bbox values"}), 400
    if w <= 0 or h <= 0:
        return jsonify({"error": "bbox has zero or negative size"}), 400

    p = _paths(request_id)
    output_dir = p['output_dir']
    cropped_dir = p['cropped_dir']
    detailed_csv = p['detailed_csv']
    summary_csv = p['summary_csv']

    edited_detailed_csv = os.path.join(output_dir, 'detailed_predictions_edited.csv')
    edited_summary_csv = os.path.join(output_dir, 'class_summary_edited.csv')
    annotations_map_edited = os.path.join(output_dir, 'annotations_map_edited.json')
    coco_json_edited = os.path.join(output_dir, 'coco_annotations_edited.json')

    # --- Load current annotation map (edited if present, else original, else empty) ---
    map_source = annotations_map_edited if os.path.exists(annotations_map_edited) else p['annotations_map']
    ann_map = {}
    if os.path.exists(map_source):
        try:
            with open(map_source, 'r', encoding='utf-8') as f:
                ann_map = json.load(f)
        except Exception as e:
            app.logger.warning(f"add_annotation: failed to read {map_source}: {e}")

    # --- Locate + crop the original full-resolution trap image ---
    orig_image_path = _find_trap_image_path(request_id)
    if not orig_image_path:
        return jsonify({"error": "original trap image not found"}), 404

    img = cv2.imread(orig_image_path)
    if img is None:
        return jsonify({"error": "failed to read original trap image"}), 500

    img_h, img_w = img.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    max_idx = 0
    for fname in ann_map.keys():
        m = re.match(r'insect_(\d+)\.jpg$', fname)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    new_filename = f"insect_{max_idx + 1}.jpg"

    os.makedirs(cropped_dir, exist_ok=True)
    crop = img[y:y + h, x:x + w]
    cv2.imwrite(os.path.join(cropped_dir, new_filename), crop)

    # --- Append row to edited detailed CSV ---
    if os.path.exists(edited_detailed_csv):
        rows = _read_detailed_csv(edited_detailed_csv)
    else:
        rows = _read_detailed_csv(detailed_csv)

    rows.append({
        'Image Name': new_filename,
        'Raw Prediction': 'Manual',
        'Confidence': '',
        'Threshold': '',
        'Final Prediction': class_name,
        'Edited_to': class_name,
    })

    fieldnames = [
        'Image Name',
        'Raw Prediction',
        'Confidence',
        'Threshold',
        'Final Prediction',
        'Edited_to'
    ]
    try:
        with open(edited_detailed_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                out = {k: r.get(k, "") for k in fieldnames}
                writer.writerow(out)
    except Exception as e:
        return jsonify({"error": "failed writing edited CSV", "detail": str(e)}), 500

    # --- Recompute summary from edited rows ---
    edited_counts = {c: 0 for c in class_names}
    for r in rows:
        final = r.get('Edited_to')
        if not final or final == 'N/A':
            final = r.get('Final Prediction')
        edited_counts[final] = edited_counts.get(final, 0) + 1

    metadata = {}
    with open(summary_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
        metadata["sample_id"] = first_row.get("sample_id", "")
        metadata["side_or_trapnum"] = first_row.get("side_or_trapnum", "")
        metadata["watershed"] = first_row.get("watershed", "")
        metadata["date"] = first_row.get("date", "")

    try:
        with open(edited_summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id','side_or_trapnum','watershed', 'date','dipteran_small', 'terrestrial_small', 'caddisfly_large', 'stonefly_large', 'mayfly_large', 'other_small'])
            writer.writerow([metadata["sample_id"], metadata["side_or_trapnum"], metadata["watershed"], metadata["date"], edited_counts.get('Dipteran', 0), edited_counts.get('Terrestrial', 0), edited_counts.get('Caddisfly', 0), edited_counts.get('Stonefly', 0), edited_counts.get('Mayfly', 0), edited_counts.get('Other', 0)])
    except Exception as e:
        return jsonify({"error": "failed writing summary", "detail": str(e)}), 500

    # --- Update annotation map + regenerate COCO ---
    ann_map[new_filename] = {
        "bbox": [x, y, w, h],
        "score": 1.0,
        "category": class_name
    }
    try:
        with open(annotations_map_edited, 'w', encoding='utf-8') as f:
            json.dump(ann_map, f, indent=2)
    except Exception as e:
        app.logger.warning(f"Failed to write annotations_map_edited.json: {e}")

    try:
        from utils.inference_pipeline import convert_to_coco
    except Exception as e:
        app.logger.warning(f"Could not import convert_to_coco: {e}")
        convert_to_coco = None

    coco_json_path = p['coco_json']
    original_fname = None
    if os.path.exists(coco_json_path):
        try:
            with open(coco_json_path, 'r', encoding='utf-8') as f:
                old = json.load(f)
            if old.get('images'):
                original_fname = old['images'][0].get('file_name')
        except Exception:
            pass
    if not original_fname:
        original_fname = os.path.basename(orig_image_path)

    if convert_to_coco:
        try:
            coco_dict = convert_to_coco(
                ann_map,
                original_filename=original_fname,
                image_path=orig_image_path
            )
            with open(coco_json_edited, 'w', encoding='utf-8') as f:
                json.dump(coco_dict, f, indent=2)
        except Exception as e:
            app.logger.warning(f"Failed to generate edited COCO JSON: {e}")

    rel = os.path.join('output', request_id, 'cropped_results', new_filename).replace('\\', '/')
    new_item = {
        "filename": new_filename,
        "raw_prediction": "Manual",
        "confidence": "",
        "final": class_name,
        "edited_to": class_name,
        "url": url_for('static', filename=rel)
    }
    edited_files_rel = {
        "summary_csv_edited": f"output/{request_id}/class_summary_edited.csv",
        "detailed_csv_edited": f"output/{request_id}/detailed_predictions_edited.csv",
        "coco_json_edited": f"output/{request_id}/coco_annotations_edited.json"
    }

    return jsonify({"ok": True, "item": new_item, "counts": edited_counts, "edited_files": edited_files_rel})


@app.route("/upload_original_to_roboflow", methods=["POST"])
def upload_original_to_roboflow():
    """
    Upload the original trap image to Roboflow, renaming in place
    so the filename matches the COCO JSON's file_name.
    """
    import os
    import json
    import shutil
    from roboflow import Roboflow

    data = request.get_json(silent=True)
    request_id = data.get("request_id") if data else None

    if not request_id:
        return jsonify({"error": "missing request_id"}), 400

    # === 1. Locate image directory ===
    trap_out_dir = os.path.join(app.root_path, 'static', 'trap_images', request_id)
    if not os.path.isdir(trap_out_dir):
        app.logger.warning("trap_images folder not found for Roboflow upload.")
        return jsonify({"error": "trap_images folder not found"}), 400

    imgs = [f for f in os.listdir(trap_out_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not imgs:
        app.logger.warning("No images found in trap_images for Roboflow upload.")
        return

    # === 2. Locate COCO annotation file ===
    annotation_path = os.path.join(app.root_path, 'static', 'output', request_id, 'coco_annotations.json')
    if not os.path.exists(annotation_path):
        app.logger.warning("COCO annotation file not found for Roboflow upload.")
        return

    # === 3. Read target filename from COCO JSON ===
    with open(annotation_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    if not coco_data.get("images"):
        app.logger.warning("No 'images' entry found in COCO JSON.")
        return

    target_filename = coco_data["images"][0]["file_name"]
    app.logger.debug(f"Target filename from COCO JSON: {target_filename}")

    # === 4. Rename actual image in place to match JSON filename ===
    current_image_path = os.path.join(trap_out_dir, imgs[0])
    target_image_path = os.path.join(trap_out_dir, target_filename)

    if current_image_path != target_image_path:
        os.rename(current_image_path, target_image_path)  # No copy, just rename
        app.logger.debug(f"Renamed image: {imgs[0]} → {target_filename}")

    # === 5. Upload to Roboflow ===
    #rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    #rf_project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)


    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    rf_project = rf.workspace("hbef-bugz").project("gcpuploadedimages")

    try:
        response = rf_project.upload(
            image_path=target_image_path,
            annotation_path=annotation_path,
            is_prediction=False,
            annotation_overwrite=True,
        )
        app.logger.info(f"Upload successful: {response}")
        monitor_roboflow_images()  # Check if we need to send an alert after upload
        return jsonify({"ok": True,})
    except Exception as e:
        app.logger.error(f"Roboflow upload failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/upload_edited_to_roboflow", methods=["POST"])
def upload_edited_to_roboflow():
    data = request.get_json(silent=True)
    request_id = data.get("request_id") if data else None

    if not request_id:
        return jsonify({"error": "missing request_id"}), 400

    trap_out_dir = os.path.join(app.root_path, 'static', 'trap_images', request_id)
    if not os.path.isdir(trap_out_dir):
        app.logger.warning("trap_images folder not found for Roboflow upload.")
        return

    imgs = [f for f in os.listdir(trap_out_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not imgs:
        app.logger.warning("No images found in trap_images for Roboflow upload.")
        return
    annotation_path = os.path.join(app.root_path, 'static', 'output', request_id, 'coco_annotations_edited.json')
    if not os.path.exists(annotation_path):
        app.logger.warning("Edited COCO annotation file not found for Roboflow upload.")
        return
    current_image_path = os.path.join(trap_out_dir, imgs[0])
    with open(annotation_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    if not coco_data.get("images"):
        app.logger.warning("No 'images' entry found in COCO JSON.")
        return
    target_filename = coco_data["images"][0]["file_name"]
    target_image_path = os.path.join(trap_out_dir, target_filename)

    if current_image_path != target_image_path:
        os.rename(current_image_path, target_image_path)  # No copy, just rename
        app.logger.debug(f"Renamed image: {imgs[0]} → {target_filename}")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    rf_project = rf.workspace("hbef-bugz").project("gcpuploadedimages")
    try:
        response = rf_project.upload(
            image_path=target_image_path,
            annotation_path=annotation_path,
            is_prediction=False,
            annotation_overwrite=True,
        )
        app.logger.info(f"Edited upload successful: {response}")
        monitor_roboflow_images()
        return jsonify({"ok": True,})
    except Exception as e:
        app.logger.error(f"Roboflow edited upload failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

# -------------------------------
# Slack alert sending function
# -------------------------------
def send_slack_alert(webhook_url, message):
    """Send a message to Slack via webhook."""
    import requests
    payload = {"text": message}
    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code != 200:
            app.logger.error(f"Slack alert failed with status {response.status_code}: {response.text}")
        else:
            app.logger.info("Slack alert sent successfully.")
    except Exception as e:
        app.logger.error(f"Error sending Slack alert: {e}")

# -------------------------------
# Roboflow monitoring function
# -------------------------------
def monitor_roboflow_images():
    """Check Roboflow dataset image count and send alert if threshold exceeded."""
    # Load threshold from file or set default
    if os.path.exists(THRESHOLD_FILE):
        threshold = json.load(open(THRESHOLD_FILE)).get("threshold", START)
    else:
        threshold = START

    # Connect to Roboflow
    #rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    #project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    
    #rf = Roboflow(api_key="TS1niacLXvvWTierCKCT")
    #rf_project = rf.workspace("insectai").project("results_test-9n8mo")
    #info = rf_project.info()
    #count = info.get("images", 0)

    import requests

    API_KEY = ROBOFLOW_API_KEY
    WORKSPACE = "hbef-bugz"
    PROJECT = "gcpuploadedimages"

    url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}?api_key={API_KEY}"
    response = requests.get(url)
    data = response.json()

    count = data["project"]["images"]

    print(f"📊 Images: {count}, Threshold: {threshold}")

    # If threshold exceeded → send email + update threshold
    if count >= threshold:
        subject = f"Roboflow Alert: {count} images"
        body = f"Dataset has {count} images (threshold {threshold} exceeded). Next: {threshold+STEP}"

        send_slack_alert(SLACK_WEBHOOK_URL, body)

        json.dump({"threshold": threshold + STEP}, open(THRESHOLD_FILE, "w"))
        print(f"✅ Threshold updated to {threshold + STEP}")



