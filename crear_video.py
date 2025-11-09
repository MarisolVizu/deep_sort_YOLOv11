import cv2
import os
import natsort

# Ruta de la carpeta con las imágenes
folder_path = r"D:\Proyectos\DEEPSORT_V2\data"

# Obtener lista de archivos de imagen y ordenarlos naturalmente
images = [img for img in os.listdir(folder_path) if img.endswith(".png")]
images = natsort.natsorted(images)

# Leer la primera imagen para obtener dimensiones
frame = cv2.imread(os.path.join(folder_path, images[0]))
height, width, layers = frame.shape

# Crear el objeto VideoWriter
video_name = "estela.mp4"
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

# Escribir cada imagen como un frame
for image in images:
    img_path = os.path.join(folder_path, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Liberar recursos
video.release()
cv2.destroyAllWindows()

print("✅ Video generado correctamente:", video_name)
