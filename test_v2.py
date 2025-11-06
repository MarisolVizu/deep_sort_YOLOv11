# ======================================================
# YOLO + DeepSORT — Rastreo de vehículos con IDs únicos
# ======================================================

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import random

# ========== CONFIGURACIÓN ==========
VIDEO_PATH = "video_generado11.avi"
YOLO_MODEL = "yolo11x.pt"  # o yolov8x.pt si lo prefieres
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inicializar YOLO
model = YOLO(YOLO_MODEL)
print(f"[INFO] Modelo YOLO cargado: {YOLO_MODEL} ({DEVICE})")

# ========== INICIALIZAR DeepSORT ==========
cfg = get_config()
cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

deepsort = DeepSort(
    cfg.DEEPSORT.REID_CKPT,
    max_dist=cfg.DEEPSORT.MAX_DIST,
    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg.DEEPSORT.MAX_AGE,
    n_init=cfg.DEEPSORT.N_INIT,
    nn_budget=cfg.DEEPSORT.NN_BUDGET,
    use_cuda=True if DEVICE == "cuda" else False,
)
print("[INFO] DeepSORT inicializado correctamente.")

# ========== FUNCIONES AUXILIARES ==========

def UI_box(x, img, color=None, label=None, line_thickness=2):
    """Dibuja una caja con etiqueta"""
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    if label:
        tf = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
        cv2.rectangle(
            img,
            (c1[0], c1[1] - t_size[1] - 4),
            (c1[0] + t_size[0], c1[1]),
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            line_thickness / 3,
            [255, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def xyxy_to_xywh(x1, y1, x2, y2):
    """Convierte coordenadas xyxy a xywh"""
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w / 2
    y_c = y1 + h / 2
    return x_c, y_c, w, h


# ========== PROCESAR VIDEO ==========
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"[ERROR] No se pudo abrir el video: {VIDEO_PATH}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Inferencia YOLO (solo autos)
    results = model(frame, conf=0.25, imgsz=1280, verbose=False)
    detections = results[0]

    if detections.boxes is None or len(detections.boxes) == 0:
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = detections.boxes.xyxy.cpu().numpy()
    confs = detections.boxes.conf.cpu().numpy()
    clss = detections.boxes.cls.cpu().numpy().astype(int)

    # Filtrar solo autos (COCO class 2)
    mask = clss == 2
    boxes = boxes[mask]
    confs = confs[mask]
    clss = clss[mask]

    # Si no hay autos detectados, pasar al siguiente frame
    if len(boxes) == 0:
        continue

    xywhs = []
    for box in boxes:
        xywhs.append(xyxy_to_xywh(*box))

    xywhs = torch.Tensor(xywhs)
    confs_t = torch.Tensor(confs)

    # ===== DeepSORT actualiza tracks =====
    outputs = deepsort.update(xywhs, confs_t, clss, frame)

    # ⚠️ Evita error si no hay resultados
    # ===== Dibujar resultados =====
    # ===== Dibujar resultados =====
    if outputs is not None and len(outputs) > 0:
        for out in outputs:
            if len(out) == 6:
                x1, y1, x2, y2, track_id, cls_id = out
            elif len(out) == 5:
                x1, y1, x2, y2, track_id = out
                cls_id = 2
            else:
                continue
            # Convertimos track_id a entero seguro
            if not isinstance(track_id, int):
                track_id = int(track_id.item())
            label = f"Car {track_id}"
            UI_box((x1, y1, x2, y2), frame, label=label)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Rastreo finalizado correctamente.")
