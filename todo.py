from limpiar_ventusky import filtrar_ventusky
from limpiar_waze import procesar_waze
from final_modelo import tracking_autos
import pandas as pd
from pathlib import Path

def procesar_carpeta_base(base_path: str):
    base = Path(base_path).resolve()

    ## === 1. LIMPIEZA UNA SOLA VEZ (global) ===
    print("🧹 Ejecutando limpieza inicial de Ventusky y Waze...")
    filtrar_ventusky("data/ventusky_data.json", "data/ventusky_limpio.csv")
    procesar_waze("data/waze_data.json", "data/waze_limpio.csv")
    print("✅ Limpieza global completada.\n")

    ## === 2. Recorrer todas las carpetas dentro de la base ===
    for subcarpeta in base.iterdir():
        if subcarpeta.is_dir():
            print(f"\n🚗 Procesando {subcarpeta.name}...")

            # Rutas absolutas aseguradas
            images_folder = str(subcarpeta.resolve())
            video_output = str((subcarpeta / f"video_{subcarpeta.name}.mp4").resolve())
            csv_output = str((subcarpeta / f"data_{subcarpeta.name}.csv").resolve())

            # === Mensajes de depuración ===
            print(f"📂 Ejecutando tracking_autos con:\n"
                  f"   - Imágenes: {images_folder}\n"
                  f"   - Video:    {video_output}\n"
                  f"   - CSV:      {csv_output}")

            # === Modelo tracking autos ===
            tracking_autos(
                images_folder=images_folder,
                video_output=video_output,
                csv_output=csv_output
            )
            print("✅ Tracking completado correctamente.")

            # === 3. Agregar ID único ===
            sitio = base.name
            id_unico = f"{sitio}_{subcarpeta.name}"
            autos = pd.read_csv(csv_output)
            autos["id"] = id_unico
            autos.to_csv(csv_output, index=False)
            print(f"✅ ID '{id_unico}' agregado a {csv_output}")

            # === 4. Cruce con Waze y Ventusky ===
            ventusky = pd.read_csv("data/ventusky_limpio.csv")
            waze = pd.read_csv("data/waze_limpio.csv")

            df = autos.merge(ventusky, on="id", how="left", suffixes=("", "_ventusky"))
            df = df.merge(waze, on="id", how="left", suffixes=("", "_waze"))

            # === Guardar resultado combinado ===
            combinado_path = base / f"combinado_{subcarpeta.name}.csv"
            df.to_csv(combinado_path, index=False)
            print(f"✅ Archivo final guardado en {combinado_path}\n")

    print("\n🎯 Procesamiento completo para todas las carpetas.")


# === EJECUCIÓN ===
if __name__ == "__main__":
    procesar_carpeta_base("data/angamos/")
