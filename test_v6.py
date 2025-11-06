import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import csv
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

# ===== CONFIGURACIÓN =====
VIDEO_INPUT = "estela.mp4"
VIDEO_OUTPUT = "output_tracking_autos.mp4"  # Opcional: None para no generar
CSV_OUTPUT = "tracking_autos.csv"
CONFIDENCE_THRESHOLD = 0.02
IMAGE_SIZE = 1280
MAX_FRAMES = None  # None para procesar todo el video, o número específico

# Inicializar YOLO
model = YOLO("yolo11x.pt")

# Inicializar DeepSORT
cfg = get_config()
cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
deepsort = DeepSort(
    cfg.DEEPSORT.REID_CKPT,
    max_dist=cfg.DEEPSORT.MAX_DIST,
    min_confidence=0.0,  # ← Acepta TODO lo que YOLO ya filtró
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

# ===== ABRIR VIDEO =====
cap = cv2.VideoCapture(VIDEO_INPUT)
if not cap.isOpened():
    raise SystemExit(f"[ERROR] No se pudo abrir el video: {VIDEO_INPUT}")

# Obtener propiedades del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"📹 Video: {width}x{height} @ {fps}fps, Total frames: {total_frames}")

# ===== PREPARAR ESCRITOR DE VIDEO (OPCIONAL) =====
video_writer = None
if VIDEO_OUTPUT:
    # Para .avi usar XVID o MJPG (más compatible)
    if VIDEO_OUTPUT.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        codec_name = "XVID"
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        codec_name = "mp4v"
    
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))
    
    if video_writer.isOpened():
        print(f"💾 Video de salida: {VIDEO_OUTPUT} ({width}x{height}) - Codec: {codec_name}")
    else:
        print(f"❌ Error: No se pudo crear el video writer")
        video_writer.release()
        video_writer = None
        raise SystemExit("No se pudo inicializar el escritor de video")

# ===== PROCESAMIENTO =====
frame_idx = 0
frames_to_process = MAX_FRAMES if MAX_FRAMES else total_frames

print(f"\n🚀 Iniciando tracking de autos...")

while frame_idx < frames_to_process:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Fin del video alcanzado.")
        break
    
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
                print(f"Frame {frame_idx + 1}: {len(outputs)} autos trackeados")
                
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
                        total_cars  # ← Nueva columna
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
    
    frame_idx += 1
    
    # Progreso cada 30 frames
    if frame_idx % 30 == 0:
        print(f"Progreso: {frame_idx}/{frames_to_process} frames procesados")

# ===== LIMPIEZA =====
cap.release()
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