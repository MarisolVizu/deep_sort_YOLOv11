"""
YOLOv11x + DeepSORT — PIDL Tracker (versión fusionada)
Detecta solo autos (clase 2 COCO), asigna IDs estables,
guarda CSV con: frame, id, x1, y1, x2, y2, vehicles_in_frame.
"""

import cv2
import csv
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


# ===================== CONFIG =====================
MODEL_PATH = "yolo11x.pt"
VIDEO_PATH = "video_generado11.avi"
CONF_THRESHOLD = 0.02
SAVE_VIDEO = True
CSV_PATH = Path("pidl_data.csv")
VIDEO_OUT_PATH = Path("output_pidlv2.mp4")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== INIT =====================
print("📦 Cargando modelo YOLOv11x...")
model = YOLO(MODEL_PATH)

print("📦 Inicializando DeepSORT...")
cfg = get_config()
cfg.merge_from_file("deep_sort.yaml")

deepsort = DeepSort(
    cfg.DEEPSORT.REID_CKPT,
    max_dist=cfg.DEEPSORT.MAX_DIST,
    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg.DEEPSORT.MAX_AGE,
    n_init=cfg.DEEPSORT.N_INIT,
    nn_budget=cfg.DEEPSORT.NN_BUDGET,
    use_cuda=(DEVICE == "cuda")
)

# ===================== VIDEO =====================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"❌ No se pudo abrir el video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_OUT_PATH), fourcc, fps, (width, height))

with open(CSV_PATH, mode='w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(["frame", "id", "x1", "y1", "x2", "y2", "vehicles_in_frame"])

# ===================== LOOP =====================
frame_idx = 0
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # --- mejora visual ligera ---
    frame = cv2.detailEnhance(frame, sigma_s=12, sigma_r=0.15)
    frame = cv2.bilateralFilter(frame, 5, 40, 40)

    # --- YOLO detección ---
    results = model(frame, conf=CONF_THRESHOLD, imgsz=1280, verbose=False)
    det = results[0]

    if len(det.boxes) == 0:
        if SAVE_VIDEO:
            writer.write(frame)
        continue

    boxes = det.boxes.xyxy.cpu().numpy()
    confs = det.boxes.conf.cpu().numpy()
    clss = det.boxes.cls.cpu().numpy().astype(int)

    # --- solo autos ---
    mask = clss == 2
    boxes, confs, clss = boxes[mask], confs[mask], clss[mask]

    if len(boxes) == 0:
        if SAVE_VIDEO:
            writer.write(frame)
        continue

    # --- DeepSORT tracking ---
    xywhs = torch.Tensor([
        [(x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1]
        for x1, y1, x2, y2 in boxes
    ])
    confs_t = torch.Tensor(confs)
    outputs = deepsort.update(xywhs, confs_t, clss, frame)

    if outputs is None or len(outputs) == 0:
        if SAVE_VIDEO:
            writer.write(frame)
        continue

    # --- Dibujar y guardar ---
    vehicles_in_frame = len(outputs)
    with open(CSV_PATH, mode='a', newline='') as f:
        writer_csv = csv.writer(f)
        for det in outputs:
            if len(det) == 6:
                x1, y1, x2, y2, track_id, cls_id = det
            elif len(det) == 5:
                x1, y1, x2, y2, track_id = det
                cls_id = 2  # auto por defecto
            else:
                continue  # saltar si no coincide el formato

            writer_csv.writerow([frame_idx, int(track_id), x1, y1, x2, y2, vehicles_in_frame])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 2)
            cv2.putText(frame, f"car ID {int(track_id)}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if SAVE_VIDEO:
        writer.write(frame)

    if frame_idx % 25 == 0:
        print(f"[INFO] Frame {frame_idx} procesado...")

# ===================== END =====================
cap.release()
if SAVE_VIDEO:
    writer.release()

print(f"\n✅ CSV guardado en: {CSV_PATH}")
print(f"🎥 Video guardado en: {VIDEO_OUT_PATH}")
print(f"🕐 Tiempo total: {time.time() - start:.1f}s")
