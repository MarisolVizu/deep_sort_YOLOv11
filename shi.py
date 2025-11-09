import cv2
import numpy as np
import torch
from pathlib import Path
import csv
import os
from natsort import natsorted
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

# ==== SAHI ====
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ===== CONFIGURACIÓN =====
IMAGES_FOLDER = "data/imagenes"  # Carpeta con las imágenes
VIDEO_OUTPUT = "output_tracking_autos3n.mp4"  # Opcional: None para no generar
CSV_OUTPUT = "tracking_autos.csv"
CONFIDENCE_THRESHOLD = 0.20
IMAGE_SIZE = 1280
FPS_OUTPUT = 30  # FPS para el video de salida

# ===== INICIALIZAR SAHI + TU MODELO YOLO11x =====
print("🚀 Cargando tu modelo YOLO11x con SAHI...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",                 # ✅ usa el backend compatible con tu modelo
    model_path="yolo11n.pt",             # tu modelo personalizado
    confidence_threshold=CONFIDENCE_THRESHOLD,
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)
print("✅ Modelo YOLO11x cargado correctamente con SAHI.")git commit -m "Migrando proyecto antiguo a nuevo repositorio"


# ===== INICIALIZAR DeepSORT =====
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
    use_cuda=torch.cuda.is_available()
)

# ===== PREPARAR CSV =====
csv_data = []
csv_headers = ["frame", "track_id", "x1", "y1", "x2", "y2", "confidence", "class", "total_cars_in_frame"]

# ===== LEER IMÁGENES DE LA CARPETA =====
if not os.path.exists(IMAGES_FOLDER):
    raise SystemExit(f"[ERROR] No se encontró la carpeta: {IMAGES_FOLDER}")

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
image_files = []
for ext in image_extensions:
    image_files.extend(Path(IMAGES_FOLDER).glob(f'*{ext}'))
    image_files.extend(Path(IMAGES_FOLDER).glob(f'*{ext.upper()}'))

image_files = natsorted(image_files)

if len(image_files) == 0:
    raise SystemExit(f"[ERROR] No se encontraron imágenes en: {IMAGES_FOLDER}")

total_frames = len(image_files)
print(f"📁 Carpeta: {IMAGES_FOLDER}")
print(f"🖼️ Total de imágenes encontradas: {total_frames}")

first_frame = cv2.imread(str(image_files[0]))
if first_frame is None:
    raise SystemExit(f"[ERROR] No se pudo leer la primera imagen: {image_files[0]}")

height, width = first_frame.shape[:2]
print(f"📐 Resolución: {width}x{height}")

# ===== VIDEO DE SALIDA =====
video_writer = None
if VIDEO_OUTPUT:
    if VIDEO_OUTPUT.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        codec_name = "XVID"
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        codec_name = "mp4v"
    
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, FPS_OUTPUT, (width, height))
    
    if video_writer.isOpened():
        print(f"💾 Video de salida: {VIDEO_OUTPUT} ({width}x{height} @ {FPS_OUTPUT}fps) - Codec: {codec_name}")
    else:
        print(f"❌ Error: No se pudo crear el video writer")
        video_writer = None

# ===== PROCESAMIENTO =====
print(f"\n🚗 Iniciando tracking de autos (con SAHI slicing)...")

for frame_idx, img_path in enumerate(image_files):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[WARNING] No se pudo leer: {img_path}, saltando...")
        continue

    frame_enhanced = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR)

    # === SAHI: detección por slices ===
    sahi_result = get_sliced_prediction(
        frame_enhanced,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # === Convertir resultados SAHI a arrays numpy ===
    boxes, confs, classes = [], [], []
    for obj in sahi_result.object_prediction_list:
        x1, y1, x2, y2 = obj.bbox.to_xyxy()
        boxes.append([x1, y1, x2, y2])
        confs.append(obj.score.value)
        classes.append(obj.category.id)

    boxes = np.array(boxes)
    confs = np.array(confs)
    classes = np.array(classes)

    # === Filtrar solo autos (class 2 en COCO) ===
    mask = classes == 2
    boxes_cars = boxes[mask]
    confs_cars = confs[mask]

    if len(boxes_cars) > 0:
        # Convertir a formato DeepSORT: [x_centro, y_centro, w, h]
        xywhs = []
        for box in boxes_cars:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            x_c, y_c = x1 + w / 2, y1 + h / 2
            xywhs.append([x_c, y_c, w, h])
        xywhs = np.array(xywhs)
        cls_ids = np.array([2] * len(boxes_cars))

        # === DeepSORT tracking ===
        outputs, mask_outputs = deepsort.update(xywhs, confs_cars, cls_ids, frame_enhanced)

        if len(outputs) > 0:
            total_cars = len(outputs)
            for output in outputs:
                x1, y1, x2, y2, cls, track_id = output
                conf_idx = min(len(confs_cars) - 1, 0)
                csv_data.append([
                    frame_idx + 1,
                    int(track_id),
                    int(x1), int(y1), int(x2), int(y2),
                    float(confs_cars[conf_idx]),
                    "car",
                    total_cars
                ])

            annotated = frame_enhanced.copy()
            for output in outputs:
                x1, y1, x2, y2, cls, track_id = output
                x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
                color = tuple(int(c) for c in np.random.RandomState(int(track_id)).randint(50, 255, 3))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                label = f"ID: {track_id}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if video_writer and video_writer.isOpened():
                annotated_resized = cv2.resize(annotated, (width, height))
                video_writer.write(annotated_resized)
        else:
            if video_writer and video_writer.isOpened():
                video_writer.write(frame)
    else:
        if video_writer and video_writer.isOpened():
            video_writer.write(frame)

    if (frame_idx + 1) % 20 == 0:
        print(f"🧩 Procesadas {frame_idx + 1}/{total_frames} imágenes...")

# ===== GUARDAR CSV =====
with open(CSV_OUTPUT, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)
    writer.writerows(csv_data)

if video_writer:
    video_writer.release()

print(f"\n✅ Tracking completado con SAHI!")
print(f"📊 CSV guardado: {CSV_OUTPUT} ({len(csv_data)} detecciones)")
if VIDEO_OUTPUT:
    print(f"🎥 Video guardado: {VIDEO_OUTPUT}")

if csv_data:
    unique_ids = len(set(row[1] for row in csv_data))
    print(f"🚗 Total de autos únicos detectados: {unique_ids}")
