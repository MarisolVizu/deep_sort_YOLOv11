"""
YOLOv11x + DeepSORT Tracker PIDL (versión simplificada FINAL)
Detecta solo autos, asigna IDs estables y guarda CSV con:
frame, id, x1, y1, x2, y2, vehicles_in_frame
"""

import argparse
from pathlib import Path
import time
import csv

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# DeepSORT (versión local)
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


# -------------------------------------------------------------
# FUNCIONES
# -------------------------------------------------------------
def init_deepsort(cfg_path="deep_sort.yaml", use_cuda=True):
    cfg = get_config()
    cfg.merge_from_file(cfg_path)
    print(f"[DEBUG] Modelo ReID: {cfg.DEEPSORT.REID_CKPT}")
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=use_cuda
    )
    return deepsort


def draw_box(frame, box, track_id):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"car {int(track_id)}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# -------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -------------------------------------------------------------
def run(source, model_path, conf_threshold=0.25, save_video=True):
    base_dir = Path(".")
    csv_path = base_dir / "pidl_data.csv"
    video_out_path = base_dir / "output_with_ids.mp4"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    deepsort = init_deepsort(use_cuda=(device == "cuda"))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"❌ No se puede abrir el video {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))

    # Crear CSV
    with open(csv_path, mode='w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame", "id", "x1", "y1", "x2", "y2", "vehicles_in_frame"])

    frame_idx = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, conf=conf_threshold, verbose=False)
        res = results[0]

        if len(res.boxes) == 0:
            if writer:
                writer.write(frame)
            continue

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        # Solo autos (clase 2 del COCO)
        mask = clss == 2
        boxes, confs, clss = boxes[mask], confs[mask], clss[mask]

        if len(boxes) == 0:
            if writer:
                writer.write(frame)
            continue

        xywhs = torch.Tensor([((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                              for x1, y1, x2, y2 in boxes])
        confs = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confs, clss, frame)

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if outputs is None or len(outputs) == 0:
            if writer:
                writer.write(frame)
            continue

        vehicles_in_frame = len(outputs)

        with open(csv_path, mode='a', newline='') as f:
            writer_csv = csv.writer(f)
            for x1, y1, x2, y2, track_id, cls_id in outputs:
                writer_csv.writerow([frame_idx, int(track_id), x1, y1, x2, y2, vehicles_in_frame])
                draw_box(frame, (x1, y1, x2, y2), track_id)

        if writer:
            writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"[INFO] Frame {frame_idx} procesado...")

    cap.release()
    if writer:
        writer.release()

    print(f"\n✅ CSV guardado en: {csv_path}")
    print(f"🎥 Video guardado en: {video_out_path}")
    print(f"🕐 Tiempo total: {time.time() - start:.1f}s")


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Video de entrada (ej. test3.mp4)")
    parser.add_argument("--model", default="yolo11x.pt", help="Modelo YOLOv11 (por defecto yolo11x.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confianza mínima YOLO")
    parser.add_argument("--save-video", type=lambda x: x.lower() in ('true', '1', 'yes'), default=True)
    args = parser.parse_args()

    run(args.source, args.model, conf_threshold=args.conf, save_video=args.save_video)
