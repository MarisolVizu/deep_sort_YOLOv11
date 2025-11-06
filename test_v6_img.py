import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import csv
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
import os
from natsort import natsorted

# ===== CONFIGURACIÓN =====
IMAGES_FOLDER = "data/imagenes"  # Carpeta con las imágenes
VIDEO_OUTPUT = "output_tracking_autos.mp4"  # Opcional: None para no generar
CSV_OUTPUT = "tracking_autos.csv"
CONFIDENCE_THRESHOLD = 0.1
IMAGE_SIZE = 1280
FPS_OUTPUT = 30  # FPS para el video de salida

# Inicializar YOLO
model = YOLO("yolo11x.pt")

# Inicializar DeepSORT
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

# ===== PREPARAR CSV =====
csv_data = []
csv_headers = ["frame", "track_id", "x1", "y1", "x2", "y2", "confidence", "class", "total_cars_in_frame"]


# ===== LEER IMÁGENES DE LA CARPETA =====
if not os.path.exists(IMAGES_FOLDER):
    raise SystemExit(f"[ERROR] No se encontró la carpeta: {IMAGES_FOLDER}")

# Obtener lista de imágenes (ordenadas naturalmente)
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
image_files = []
for ext in image_extensions:
    image_files.extend(Path(IMAGES_FOLDER).glob(f'*{ext}'))
    image_files.extend(Path(IMAGES_FOLDER).glob(f'*{ext.upper()}'))

# Ordenar naturalmente (frame1.jpg, frame2.jpg, ..., frame10.jpg)
image_files = natsorted(image_files)

if len(image_files) == 0:
    raise SystemExit(f"[ERROR] No se encontraron imágenes en: {IMAGES_FOLDER}")

total_frames = len(image_files)
print(f"📁 Carpeta: {IMAGES_FOLDER}")
print(f"🖼️ Total de imágenes encontradas: {total_frames}")

# Leer primera imagen para obtener dimensiones
first_frame = cv2.imread(str(image_files[0]))
if first_frame is None:
    raise SystemExit(f"[ERROR] No se pudo leer la primera imagen: {image_files[0]}")

height, width = first_frame.shape[:2]
print(f"📐 Resolución: {width}x{height}")

# ===== PREPARAR ESCRITOR DE VIDEO (OPCIONAL) =====
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
print(f"\n🚀 Iniciando tracking de autos...")

for frame_idx, img_path in enumerate(image_files):
    # Leer imagen
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[WARNING] No se pudo leer: {img_path}, saltando...")
        continue
    
    # 🔧 Mejora de resolución (solo resize, sin filtros pesados)
    frame_enhanced = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # 🔍 Inferencia YOLO
    results = model.predict(frame_enhanced, conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE, verbose=False)
    r = results[0]
    
    # Preparar detecciones para DeepSORT
    if len(r.boxes):
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        
        # 🚗 Filtrar solo autos (class == 2 en COCO)
        mask = classes == 2
        boxes_cars = boxes[mask]
        confs_cars = confs[mask]
        
        if len(boxes_cars) > 0:
            # Convertir a formato DeepSORT: [x_centro, y_centro, w, h]
            xywhs = []
            for box in boxes_cars:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                x_c = x1 + w / 2
                y_c = y1 + h / 2
                xywhs.append([x_c, y_c, w, h])
            
            xywhs = np.array(xywhs)
            
            # Crear array de clases (todas son autos = clase 2)
            cls_ids = np.array([2] * len(boxes_cars))
            
            # 🎯 Actualizar DeepSORT
            outputs, mask_outputs = deepsort.update(xywhs, confs_cars, cls_ids, frame_enhanced)
            
            # outputs formato: [x1, y1, x2, y2, class, track_id]
            if len(outputs) > 0:
                print(f"Imagen {frame_idx + 1}/{total_frames} ({img_path.name}): {len(outputs)} autos trackeados")
                
                # Contar total de autos en este frame
                total_cars = len(outputs)
                
                # 📊 Guardar en CSV
                for output in outputs:
                    x1, y1, x2, y2, cls, track_id = output
                    # Buscar la confianza correspondiente
                    conf_idx = min(len(confs_cars) - 1, 0) if len(confs_cars) > 0 else 0
                    csv_data.append([
                        frame_idx + 1,
                        int(track_id),
                        int(x1), int(y1), int(x2), int(y2),
                        float(confs_cars[conf_idx]) if len(confs_cars) > 0 else 0.0,
                        "car",
                        total_cars
                    ])
                
                # 🔲 Dibujar en el frame
                annotated = frame_enhanced.copy()
                for output in outputs:
                    x1, y1, x2, y2, cls, track_id = output
                    x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
                    
                    # Color único por ID
                    color = tuple(int(c) for c in np.random.RandomState(int(track_id)).randint(50, 255, 3))
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    
                    # Etiqueta con ID
                    label = f"ID: {track_id}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 💾 Escribir frame al video
                if video_writer and video_writer.isOpened():
                    # El frame_enhanced es más grande, ajustar al tamaño de salida
                    annotated_resized = cv2.resize(annotated, (width, height))
                    video_writer.write(annotated_resized)
            else:
                # Sin detecciones, guardar frame original
                if video_writer and video_writer.isOpened():
                    video_writer.write(frame)
        else:
            # No hay autos detectados
            if video_writer and video_writer.isOpened():
                video_writer.write(frame)
    else:
        # No hay detecciones en absoluto
        if video_writer and video_writer.isOpened():
            video_writer.write(frame)
    
    # Progreso cada 30 frames
    if (frame_idx + 1) % 30 == 0:
        print(f"Progreso: {frame_idx + 1}/{total_frames} imágenes procesadas")

# ===== LIMPIEZA =====
if video_writer:
    video_writer.release()

# ===== GUARDAR CSV =====
with open(CSV_OUTPUT, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)
    writer.writerows(csv_data)

print(f"\n✅ Tracking completado!")
print(f"📊 CSV guardado: {CSV_OUTPUT} ({len(csv_data)} detecciones)")
if VIDEO_OUTPUT:
    print(f"🎥 Video guardado: {VIDEO_OUTPUT}")

# ===== ESTADÍSTICAS =====
if csv_data:
    unique_ids = len(set(row[1] for row in csv_data))
    print(f"🚗 Total de autos únicos detectados: {unique_ids}")