from limpiar_ventusky import filtrar_ventusky
from limpiar_waze import procesar_waze
from tracking_juliov2 import tracking_autos
import pandas as pd
from pathlib import Path


# === FUNCIÓN 1: Generar CSV del timestamp específico ===
def generar_csv_tracking_timestamp(base_path: str):
    base = Path(base_path).resolve()
    print(f"🧹 Ejecutando limpieza inicial en {base.name}...")

    # Limpieza global
    filtrar_ventusky("data/ventusky_data.json", "data/ventusky_limpio.csv")
    procesar_waze("data/waze_data.json", "data/waze_limpio.csv")
    print("✅ Limpieza de Ventusky y Waze completada.\n")

    # --- Procesamiento del timestamp ---
    sitio = base.parent.name
    carpeta_timestamp = base.name

    print(f"\n📂 Procesando timestamp {carpeta_timestamp} del sitio {sitio}...")

    images_folder = str(base.resolve())
    video_output = str((base / f"video_{carpeta_timestamp}.mp4").resolve())
    csv_output = str((base / f"data_{carpeta_timestamp}.csv").resolve())

    # Ejecuta el tracking
    tracking_autos(
        images_folder=images_folder,
        video_output=video_output,
        csv_output=csv_output
    )
    print("🚗 Tracking completado correctamente.")

    # Leer CSV generado
    autos = pd.read_csv(csv_output)

    # Añadir ID único
    id_unico = f"{sitio}_{carpeta_timestamp}"
    autos["id"] = id_unico

    # fps fijo o leído si existe
    if "fps_promedio" not in autos.columns:
        autos["fps_promedio"] = 30

    # Guardar CSV actualizado
    autos.to_csv(csv_output, index=False)
    print(f"🆔 ID '{id_unico}' y columnas adicionales agregadas a {csv_output}")
    print("\n✅ CSV generado correctamente para el timestamp.")


# === FUNCIÓN 2: Cruce tolerante ±1 hora ===
def cruzar_datos(base_path: str):
    base = Path(base_path).resolve()
    ventusky = pd.read_csv("data/ventusky_limpio.csv")
    waze = pd.read_csv("data/waze_limpio.csv")

    # --- Extraer partes del ID ---
    def extraer_partes(id_str):
        partes = str(id_str).split("_")
        if len(partes) < 3:
            return None, None, None
        sitio, fecha, hora = partes[0], partes[1], partes[2].split("-")[0]
        try:
            hora_int = int(hora)
        except ValueError:
            hora_int = None
        return sitio, fecha, hora_int

    for df in [ventusky, waze]:
        df[["sitio", "fecha", "hora"]] = df["id"].apply(
            lambda x: pd.Series(extraer_partes(x))
        )

    print(f"\n📊 Cruzando datos para {base.name}...")

    csv_path = base / f"data_{base.name}.csv"
    if not csv_path.exists():
        print(f"⚠️ No se encontró {csv_path}, se omite.")
        return

    autos = pd.read_csv(csv_path)
    autos[["sitio", "fecha", "hora"]] = autos["id"].apply(
        lambda x: pd.Series(extraer_partes(x))
    )

    def buscar_por_rango(df_ref, sitio, fecha, hora):
        if hora is None:
            return None
        rango = [hora - 1, hora, hora + 1]
        return df_ref[
            (df_ref["sitio"] == sitio)
            & (df_ref["fecha"] == fecha)
            & (df_ref["hora"].isin(rango))
        ]

    resultados = []
    for _, row in autos.iterrows():
        sitio, fecha, hora = row["sitio"], row["fecha"], row["hora"]
        vent_row = buscar_por_rango(ventusky, sitio, fecha, hora)
        waze_row = buscar_por_rango(waze, sitio, fecha, hora)
        vent_data = vent_row.iloc[0].to_dict() if not vent_row.empty else {}
        waze_data = waze_row.iloc[0].to_dict() if not waze_row.empty else {}
        combinado = {**row.to_dict(), **vent_data, **waze_data}
        resultados.append(combinado)

    df_final = pd.DataFrame(resultados)
    combinado_path = base / f"combinado_{base.name}.csv"
    df_final.to_csv(combinado_path, index=False)
    print(f"\n✅ Archivo combinado guardado en {combinado_path}")
    print("🌎 Cruce con tolerancia ±1 hora completado.")


# === EJECUCIÓN ===
if __name__ == "__main__":
    generar_csv_tracking_timestamp("data/angamos/2025-10-24_15-54-13/") ## 2025-10-23_15-45-48/")
    cruzar_datos("data/angamos/2025-10-24_15-54-13")
