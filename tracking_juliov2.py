import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import csv
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
import os
from natsort import natsorted
from datetime import datetime
import time

def tracking_autos(
    images_folder: str,
    video_output: str = None,
    csv_output: str = "tracking_autos.csv",
    model_path: str = "yolo11x.pt",
    conf_threshold: float = 0.2,
    image_size: int = 1280,
    fps_output: int = 30
):
    """
    Realiza el tracking de autos procesando imágenes organizadas en carpetas tipo:
    avenida_1/1/, avenida_1/2/, avenida_2/1/, etc.
    Guarda un video anotado y un CSV consolidado con todos los frames.
    """

    base = Path(images_folder).resolve()
    print(f"📂 Carpeta base: {base}")

    # === 1. Construir lista ordenada de imágenes (manteniendo tu lógica)
    avenidas = sorted([p for p in base.iterdir() if p.is_dir() and "avenida" in p.name])
    all_images = []
    subdirs = sorted({int(p.name) for a in avenidas for p in a.iterdir() if p.is_dir() and p.name.isdigit()})

    for subnum in subdirs:
        for avenida in avenidas:
            carpeta = avenida / str(subnum)
            if carpeta.exists():
                imgs = sorted(list(carpeta.glob("*.png")) + list(carpeta.glob("*.jpg")))
                all_images.extend(imgs)

    if not all_images:
        print("⚠️ No se encontraron imágenes para procesar.")
        return

    print(f"🚗 Total de imágenes encontradas: {len(all_images)}")

    # === 2. Cargar modelo YOLO ===
    print(f"🔹 Cargando modelo YOLO desde {model_path} ...")
    model = YOLO(model_path)

    # === 3. Configurar DeepSort ===
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

    # === 4. Configurar video de salida ===
    first_frame = cv2.imread(str(all_images[0]))
    if first_frame is None:
        raise SystemExit("❌ No se pudo leer la primera imagen.")
    height, width = first_frame.shape[:2]

    video_writer = None
    if video_output:
        fourcc = cv2.VideoWriter_fourcc(*("mp4v"))
        video_writer = cv2.VideoWriter(video_output, fourcc, fps_output, (width, height))
        print(f"🎥 Video de salida: {video_output}")

    # === 5. Procesamiento frame a frame ===
    csv_data = []
    start_time = time.time()

    for frame_idx, img_path in enumerate(all_images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[⚠️] No se pudo leer: {img_path}, saltando...")
            continue

        direccion = img_path.parts[-3] if "avenida" in img_path.parts[-3] else "N/A"
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
                xywhs = np.array([[x1 + (x2 - x1)/2, y1 + (y2 - y1)/2, x2 - x1, y2 - y1]
                                  for x1, y1, x2, y2 in boxes_cars])
                cls_ids = np.array([2] * len(boxes_cars))
                outputs, _ = deepsort.update(xywhs, confs_cars, cls_ids, frame_enhanced)

                if len(outputs) > 0:
                    total_cars = len(outputs)
                    for output in outputs:
                        x1, y1, x2, y2, cls, track_id = output
                        csv_data.append([
                            frame_idx + 1,
                            int(track_id),
                            int(x1), int(y1), int(x2), int(y2),
                            float(np.mean(confs_cars)),
                            "car",
                            total_cars,
                            direccion,
                            fecha, hora
                        ])

                    annotated = frame_enhanced.copy()
                    for x1, y1, x2, y2, cls, track_id in outputs:
                        color = tuple(int(c) for c in np.random.RandomState(track_id).randint(50, 255, 3))
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                        label = f"ID:{int(track_id)}"
                        cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    if video_writer:
                        video_writer.write(cv2.resize(annotated, (width, height)))
        else:
            if video_writer:
                video_writer.write(frame)

        if (frame_idx + 1) % 20 == 0:
            print(f"⏳ Procesadas {frame_idx + 1}/{len(all_images)} imágenes...")

    # === 6. Guardar CSV ===
    with open(csv_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "track_id", "x1", "y1", "x2", "y2",
            "confidence", "class", "total_cars_in_frame",
            "direccion", "fecha", "hora"
        ])
        writer.writerows(csv_data)

    if video_writer:
        video_writer.release()

    elapsed = round(time.time() - start_time, 2)
    print(f"\n✅ Tracking completado en {elapsed}s")
    print(f"📄 CSV generado: {csv_output} ({len(csv_data)} registros)")
    if video_output:
        print(f"🎥 Video generado: {video_output}")
    if csv_data:
        print(f"🚘 Autos únicos detectados: {len(set(r[1] for r in csv_data))}")
