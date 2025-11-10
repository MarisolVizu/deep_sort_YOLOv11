import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import csv
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
import os
from natsort import natsorted
from datetime import datetime  # ✅ nuevo

def tracking_autos(
    images_folder: str,
    video_output: str = None,
    csv_output: str = "tracking_autos.csv",
    model_path: str = "yolo11n.pt",
    conf_threshold: float = 0.1,
    image_size: int = 1280,
    fps_output: int = 30
):
    """
    Realiza el seguimiento (tracking) de autos a partir de una secuencia de imágenes.
    Guarda opcionalmente un video con las anotaciones y un CSV con las coordenadas e IDs.
    """

    # ===== CARGAR MODELO YOLO =====
    print("🔹 Cargando modelo YOLO...")
    model = YOLO(model_path)

    # ===== CONFIGURAR DEEPSORT =====
    cfg = get_config()
    cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=0.0,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=True
    )

    # ===== LEER IMÁGENES =====
    if not os.path.exists(images_folder):
        raise SystemExit(f"[ERROR] No se encontró la carpeta: {images_folder}")

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_folder).glob(f'*{ext}'))
        image_files.extend(Path(images_folder).glob(f'*{ext.upper()}'))

    image_files = natsorted(image_files)
    if not image_files:
        raise SystemExit(f"[ERROR] No se encontraron imágenes en: {images_folder}")

    total_frames = len(image_files)
    print(f"📁 Carpeta: {images_folder}")
    print(f"🖼️ Total de imágenes: {total_frames}")

    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        raise SystemExit(f"[ERROR] No se pudo leer la primera imagen: {image_files[0]}")

    height, width = first_frame.shape[:2]
    print(f"📐 Resolución base: {width}x{height}")

    # ===== CONFIGURAR VIDEO OUTPUT =====
    video_writer = None
    if video_output:
        fourcc = cv2.VideoWriter_fourcc(*('XVID' if video_output.endswith('.avi') else 'mp4v'))
        video_writer = cv2.VideoWriter(video_output, fourcc, fps_output, (width, height))
        if video_writer.isOpened():
            print(f"💾 Video de salida: {video_output} ({width}x{height} @ {fps_output}fps)")
        else:
            print(f"❌ No se pudo crear el video writer")
            video_writer = None

    # ===== PROCESAMIENTO =====
    print("\n🚀 Iniciando tracking de autos...")

    csv_data = []
    csv_headers = [
        "frame", "track_id", "x1", "y1", "x2", "y2",
        "confidence", "class", "total_cars_in_frame",
        "fecha", "hora"  # ✅ nuevas columnas
    ]

    for frame_idx, img_path in enumerate(image_files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARNING] No se pudo leer: {img_path}, saltando...")
            continue

        # Fecha y hora actual del procesamiento (puede ser la del nombre si prefieres)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fecha, hora = timestamp.split("_")

        frame_enhanced = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        results = model.predict(frame_enhanced, conf=conf_threshold, imgsz=image_size, verbose=False)
        r = results[0]

        if len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            mask = classes == 2  # autos
            boxes_cars, confs_cars = boxes[mask], confs[mask]

            if len(boxes_cars) > 0:
                xywhs = np.array([
                    [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, x2 - x1, y2 - y1]
                    for x1, y1, x2, y2 in boxes_cars
                ])
                cls_ids = np.array([2] * len(boxes_cars))
                outputs, _ = deepsort.update(xywhs, confs_cars, cls_ids, frame_enhanced)

                if len(outputs) > 0:
                    total_cars = len(outputs)
                    for output in outputs:
                        x1, y1, x2, y2, cls, track_id = output
                        csv_data.append([
                            frame_idx + 1, int(track_id),
                            int(x1), int(y1), int(x2), int(y2),
                            float(np.mean(confs_cars)),
                            "car", total_cars,
                            fecha, hora  # ✅ agregar aquí
                        ])

                    annotated = frame_enhanced.copy()
                    for x1, y1, x2, y2, cls, track_id in outputs:
                        x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
                        color = tuple(int(c) for c in np.random.RandomState(track_id).randint(50, 255, 3))
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                        label = f"ID: {track_id}"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
                        cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    if video_writer and video_writer.isOpened():
                        video_writer.write(cv2.resize(annotated, (width, height)))
        else:
            if video_writer and video_writer.isOpened():
                video_writer.write(frame)

        if (frame_idx + 1) % 30 == 0:
            print(f"Progreso: {frame_idx + 1}/{total_frames}")

    # ===== GUARDAR CSV =====
    with open(csv_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(csv_data)

    if video_writer:
        video_writer.release()

    print(f"\n✅ Tracking completado!")
    print(f"📊 CSV guardado: {csv_output} ({len(csv_data)} registros)")
    if video_output:
        print(f"🎥 Video guardado: {video_output}")

    if csv_data:
        unique_ids = len(set(row[1] for row in csv_data))
        print(f"🚗 Autos únicos detectados: {unique_ids}")
