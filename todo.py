from limpiar_ventusky import filtrar_ventusky
from limpiar_waze import procesar_waze
from final_modelo import tracking_autos
import pandas as pd
from pathlib import Path

## === 1. LIMPIEZA DE UNA VEZ ===
filtrar_ventusky("data/ventusky_data.json", "data/ventusky_limpio.csv")
procesar_waze("data/waze_data.json", "data/waze_limpio.csv")

## === 2. MODELO (tracking autos) ===
images_folder = "data/angamos/2025-10-13_12-32-14"
video_output = "data/angamos/video_2025-10-13_12-32-14.mp4"
csv_output = "data/angamos/data_2025-10-13_12-32-14.csv"

tracking_autos(images_folder=images_folder, video_output=video_output, csv_output=csv_output)

## === 3. AGREGAR ID ÚNICO A CSV DE AUTOS ===
carpeta = Path(images_folder)
sitio = carpeta.parent.name             # -> "angamos"
subcarpeta = carpeta.name               # -> "2025-10-13_12-32-14"
id_unico = f"{sitio}_{subcarpeta}"      # -> "angamos_2025-10-13_12-32-14"

autos = pd.read_csv(csv_output)
autos["id"] = id_unico                  # mismo valor para todo el CSV
autos.to_csv(csv_output, index=False)
print(f"✅ ID '{id_unico}' agregado a {csv_output}")

## === 4. CRUCE CON WAZE Y VENTUSKY ===
ventusky = pd.read_csv("data/ventusky_limpio.csv")
waze = pd.read_csv("data/waze_limpio.csv")

# left join del CSV de autos con los datos externos
df = autos.merge(ventusky, on="id", how="left", suffixes=("", "_ventusky"))
df = df.merge(waze, on="id", how="left", suffixes=("", "_waze"))

# guardar resultado combinado
df.to_csv(f"data/{sitio}/combinado_{subcarpeta}.csv", index=False)
print(f"✅ Archivo final guardado en data/{sitio}/combinado_{subcarpeta}.csv")
