"""Auto-label custom images using YOLO (YOLO11 / YOLOv8) and export to CVAT formats.

Supports:
- YOLO11 (yolo11n.pt, yolo11x.pt) or YOLOv8 (yolov8n.pt, yolov8x.pt)
- Exports:
  1. CVAT for images 1.1 (annotations.xml) - Direct XML upload to CVAT
  2. YOLO 1.1 (cvat_yolo_upload.zip) - Standard YOLO zip upload to CVAT
  3. COCO 1.0 (instances_default.json) - Standard COCO json upload to CVAT
  4. Visual preview (visualized/*.jpg) - Local verification before uploading

Usage:
    python tools/auto_label_for_cvat.py --input data/image --conf 0.3
    python tools/auto_label_for_cvat.py --model ../yolov8x.pt --conf 0.25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.dom.minidom
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Auto-detect ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("[-] Thiếu thư viện 'ultralytics'. Vui lòng cài đặt: pip install ultralytics")
    sys.exit(1)


def find_default_model() -> str:
    """Find available YOLO model, prioritizing YOLO11, then fallback to YOLOv8."""
    search_dirs = [
        Path("models"),
        Path("."),
        Path(".."),
        Path(__file__).resolve().parents[2],  # vin_ai_in_action root
    ]
    candidates = [
        "yolo11x.pt",
        "yolo11m.pt",
        "yolo11s.pt",
        "yolo11n.pt",
        "yolov8x.pt",
        "yolov8n.pt",
    ]
    for d in search_dirs:
        for c in candidates:
            p = d / c
            if p.exists():
                return str(p.resolve())

    return "yolo11n.pt"


def run_auto_label(
    input_dir: Path,
    output_dir: Path,
    model_path: str,
    conf_thresh: float = 0.3,
    iou_thresh: float = 0.45,
    save_visual: bool = True,
) -> None:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    print(f"[+] Đang tải mô hình YOLO từ: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[-] Lỗi tải mô hình {model_path}: {e}")
        fallback = Path("..") / "yolov8x.pt"
        if fallback.exists() and str(fallback.resolve()) != model_path:
            print(f"[!] Thử chuyển sang mô hình có sẵn: {fallback}")
            model = YOLO(str(fallback.resolve()))
        else:
            raise

    # 2. Get list of images
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in supported_exts])
    if not image_paths:
        print(f"[-] Không tìm thấy ảnh nào trong: {input_dir}")
        return

    print(f"[+] Tìm thấy {len(image_paths)} ảnh cần gán nhãn: {[p.name for p in image_paths]}")

    # Prepare structures for CVAT XML, COCO JSON, and YOLO TXT
    xml_root = ET.Element("annotations")
    ET.SubElement(xml_root, "version").text = "1.1"
    xml_meta = ET.SubElement(xml_root, "meta")
    xml_task = ET.SubElement(xml_meta, "task")
    ET.SubElement(xml_task, "name").text = "auto_labeled_task"
    xml_labels = ET.SubElement(xml_task, "labels")

    coco_categories = []
    category_map = {}  # class_id -> coco_cat_id
    used_labels = set()

    # Pre-populate all model classes into XML task metadata
    for cid, name in model.names.items():
        xml_label = ET.SubElement(xml_labels, "label")
        ET.SubElement(xml_label, "name").text = str(name)
        ET.SubElement(xml_label, "type").text = "rectangle"

    yolo_dir = output_dir / "yolo_data"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    visual_dir = output_dir / "visualized"
    if save_visual:
        visual_dir.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    ann_id_counter = 1

    total_boxes = 0

    # 3. Predict & Process Each Image
    for img_idx, img_path in enumerate(image_paths):
        with Image.open(img_path) as im:
            width, height = im.size
            im_draw = im.copy() if save_visual else None

        draw = ImageDraw.Draw(im_draw) if im_draw else None

        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
        )

        r = results[0]
        boxes = r.boxes

        # XML image node
        xml_image = ET.SubElement(
            xml_root,
            "image",
            id=str(img_idx),
            name=img_path.name,
            width=str(width),
            height=str(height),
        )

        # COCO image entry
        coco_images.append({
            "id": img_idx + 1,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        # YOLO txt lines
        yolo_lines = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())

                # Coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center, y_center, bw, bh = box.xywhn[0].tolist()

                # Clamp values
                x1 = max(0.0, min(float(width), x1))
                y1 = max(0.0, min(float(height), y1))
                x2 = max(0.0, min(float(width), x2))
                y2 = max(0.0, min(float(height), y2))

                w_box = x2 - x1
                h_box = y2 - y1
                if w_box <= 0 or h_box <= 0:
                    continue

                total_boxes += 1
                used_labels.add(cls_name)

                # 1. CVAT XML box
                xml_box = ET.SubElement(
                    xml_image,
                    "box",
                    label=cls_name,
                    occluded="0",
                    source="auto_yolo",
                    xtl=f"{x1:.2f}",
                    ytl=f"{y1:.2f}",
                    xbr=f"{x2:.2f}",
                    ybr=f"{y2:.2f}",
                    z_order="0",
                )
                ET.SubElement(xml_box, "attribute", name="confidence").text = f"{conf:.3f}"

                # 2. YOLO TXT line
                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

                # 3. COCO annotation
                if cls_id not in category_map:
                    category_map[cls_id] = len(coco_categories) + 1
                    coco_categories.append({
                        "id": category_map[cls_id],
                        "name": cls_name,
                        "supercategory": "object",
                    })

                coco_annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_idx + 1,
                    "category_id": category_map[cls_id],
                    "bbox": [round(x1, 2), round(y1, 2), round(w_box, 2), round(h_box, 2)],
                    "area": round(w_box * h_box, 2),
                    "segmentation": [],
                    "iscrowd": 0,
                    "score": round(conf, 3),
                })
                ann_id_counter += 1

                # 4. Visual draw
                if draw:
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    draw.text((x1 + 2, max(0, y1 - 12)), f"{cls_name} {conf:.2f}", fill="yellow")

        # Save YOLO txt
        txt_path = yolo_dir / f"{img_path.stem}.txt"
        txt_path.write_text("\n".join(yolo_lines), encoding="utf-8")

        # Save Visualized image
        if im_draw:
            im_draw.save(visual_dir / f"labeled_{img_path.name}")

    # 4. Export CVAT XML format
    xml_str = ET.tostring(xml_root, encoding="utf-8")
    parsed_xml = xml.dom.minidom.parseString(xml_str)
    pretty_xml = parsed_xml.toprettyxml(indent="  ", encoding="utf-8")
    xml_file = output_dir / "annotations.xml"
    xml_file.write_bytes(pretty_xml)
    print(f"[✓] Đã tạo file CVAT XML: {xml_file}")

    # 5. Export COCO format
    coco_data = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }
    coco_file = output_dir / "instances_default.json"
    coco_file.write_text(json.dumps(coco_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[✓] Đã tạo file COCO JSON: {coco_file}")

    # 6. Export YOLO 1.1 format zip (CVAT strict standard)
    obj_names_file = yolo_dir / "obj.names"
    names_list = [model.names[i] for i in range(len(model.names))]
    obj_names_file.write_text("\n".join(names_list), encoding="utf-8")

    # obj.data required strictly by CVAT YOLO 1.1 importer
    obj_data_file = yolo_dir / "obj.data"
    obj_data_content = (
        f"classes = {len(names_list)}\n"
        f"train = data/train.txt\n"
        f"names = data/obj.names\n"
        f"backup = backup/\n"
    )
    obj_data_file.write_text(obj_data_content, encoding="utf-8")

    # train.txt listing images
    train_txt_file = yolo_dir / "train.txt"
    train_txt_lines = [f"data/obj_train_data/{p.name}" for p in image_paths]
    train_txt_file.write_text("\n".join(train_txt_lines), encoding="utf-8")

    yolo_zip_path = output_dir / "cvat_yolo_upload.zip"
    with zipfile.ZipFile(yolo_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(obj_data_file, arcname="obj.data")
        zf.write(obj_names_file, arcname="obj.names")
        zf.write(train_txt_file, arcname="train.txt")
        # CVAT also looks for names at data/obj.names or root
        zf.write(obj_names_file, arcname="data/obj.names")
        zf.write(train_txt_file, arcname="data/train.txt")
        for txt_f in yolo_dir.glob("*.txt"):
            if txt_f.name not in ("train.txt",):
                zf.write(txt_f, arcname=f"obj_train_data/{txt_f.name}")
                zf.write(txt_f, arcname=f"data/obj_train_data/{txt_f.name}")
    print(f"[✓] Đã đóng gói ZIP định dạng YOLO 1.1 chuẩn CVAT: {yolo_zip_path}")

    # 7. Summary
    labels_summary_file = output_dir / "detected_labels.json"
    labels_summary_file.write_text(json.dumps(list(used_labels), indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("HOÀN THÀNH GÁN NHÃN SƠ BỘ CHO CVAT:")
    print(f" - Tổng số ảnh xử lý : {len(image_paths)}")
    print(f" - Tổng số box tìm thấy: {total_boxes}")
    print(f" - Các nhãn xuất hiện  : {sorted(list(used_labels))}")
    print("\nCÁCH IMPORT LÊN CVAT:")
    print(" Cách 1 (Khuyên dùng - Nhanh nhất):")
    print(f"   Vào Task trên CVAT -> 'Actions' -> 'Upload Annotations' -> Chọn 'CVAT for images 1.1' -> Chọn file:")
    print(f"   --> {xml_file}")
    print(" Cách 2 (Định dạng YOLO 1.1):")
    print(f"   Vào Task trên CVAT -> 'Actions' -> 'Upload Annotations' -> Chọn 'YOLO 1.1' -> Chọn file ZIP:")
    print(f"   --> {yolo_zip_path}")
    print(" Cách 3 (Định dạng COCO 1.0):")
    print(f"   Vào Task trên CVAT -> 'Actions' -> 'Upload Annotations' -> Chọn 'COCO 1.0' -> Chọn file:")
    print(f"   --> {coco_file}")
    print(f"\nXem trước ảnh đã vẽ box tại: {visual_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label images using YOLO for CVAT upload")
    parser.add_argument("--input", type=str, default="data/image", help="Folder containing images")
    parser.add_argument("--output", type=str, default="data/cvat_export", help="Output folder")
    parser.add_argument("--model", type=str, default="", help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")

    args = parser.parse_args()

    model_to_use = args.model if args.model else find_default_model()
    run_auto_label(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        model_path=model_to_use,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    )
