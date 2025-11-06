# test_yolo_saveframes_cars.py
import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np




# ===== CONFIGURACIÓN =====
model = YOLO("yolo11x.pt")  # o "yolov8x.pt" si lo tienes
video_path = "video_generado11.avi"
output_dir = Path("frames_detectados_autos")
output_dir.mkdir(exist_ok=True)

# ===== PROCESAMIENTO =====
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit(f"[ERROR] No se pudo abrir el video: {video_path}")

i = 0
while i < 50:  # procesa los primeros 50 frames
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Fin del video.")
        break

    # 🔧 Mejora visual sin .pb (realce + upscale)
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    frame = cv2.detailEnhance(frame, sigma_s=12, sigma_r=0.15)
    frame = cv2.bilateralFilter(frame, 5, 40, 40)
    
    # 🔍 Inferencia YOLO optimizada
    res = model.predict(frame, conf=0.02, imgsz=1280, verbose=False)
    r = res[0]

    if len(r.boxes):
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        # 🚗 Filtrar solo autos (class == 2 en COCO)
        mask = classes == 2
        boxes, classes, confs = boxes[mask], classes[mask], confs[mask]
    else:
        boxes, classes, confs = np.array([]), np.array([]), np.array([])

    print(f"Frame {i+1}: autos detectados={len(boxes)}")

    # 🔲 Dibujar solo autos
    annotated = frame.copy()
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            annotated,
            f"car {conf:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # 💾 Guardar frame anotado
    output_path = output_dir / f"frame_{i+1:03d}.jpg"
    cv2.imwrite(str(output_path), annotated)

    i += 1

cap.release()
print(f"\n✅ Se guardaron {i} frames con solo autos en '{output_dir}'")
