from limpiar_ventusky import filtrar_ventusky
from limpiar_waze import procesar_waze
from final_modelo import tracking_autos
import pandas as pd
from pathlib import Path


# === FUNCIÓN 1: Generar CSVs individuales ===
def generar_csvs_tracking(base_path: str):
    base = Path(base_path).resolve()

    print("🧹 Ejecutando limpieza inicial de Ventusky y Waze...")
    filtrar_ventusky("data/ventusky_data.json", "data/ventusky_limpio.csv")
    procesar_waze("data/waze_data.json", "data/waze_limpio.csv")
    print("✅ Limpieza global completada.\n")

    for subcarpeta in base.iterdir():
        if subcarpeta.is_dir():
            print(f"\n🚗 Procesando {subcarpeta.name}...")

            images_folder = str(subcarpeta.resolve())
            video_output = str((subcarpeta / f"video_{subcarpeta.name}.mp4").resolve())
            csv_output = str((subcarpeta / f"data_{subcarpeta.name}.csv").resolve())

            tracking_autos(
                images_folder=images_folder,
                video_output=video_output,
                csv_output=csv_output
            )
            print("✅ Tracking completado correctamente.")

            sitio = base.name
            id_unico = f"{sitio}_{subcarpeta.name}"
            autos = pd.read_csv(csv_output)
            autos["id"] = id_unico
            autos.to_csv(csv_output, index=False)
            print(f"✅ ID '{id_unico}' agregado a {csv_output}")

    print("\n🎯 CSVs generados correctamente para todas las carpetas.")


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

    # Crear columnas auxiliares para cada dataset
    for df in [ventusky, waze]:
        df[["sitio", "fecha", "hora"]] = df["id"].apply(
            lambda x: pd.Series(extraer_partes(x))
        )

    for subcarpeta in base.iterdir():
        if subcarpeta.is_dir():
            print(f"\n🔗 Cruzando datos de {subcarpeta.name}...")

            csv_path = subcarpeta / f"data_{subcarpeta.name}.csv"
            if not csv_path.exists():
                print(f"⚠️ No se encontró {csv_path}, se omite.")
                continue

            autos = pd.read_csv(csv_path)
            autos[["sitio", "fecha", "hora"]] = autos["id"].apply(
                lambda x: pd.Series(extraer_partes(x))
            )

            # --- Función auxiliar para buscar coincidencias ±1 hora ---
            def buscar_por_rango(df_ref, sitio, fecha, hora):
                if hora is None:
                    return None
                rango = [hora - 1, hora, hora + 1]
                return df_ref[
                    (df_ref["sitio"] == sitio)
                    & (df_ref["fecha"] == fecha)
                    & (df_ref["hora"].isin(rango))
                ]

            # --- Cruce manual por cada registro ---
            resultados = []
            for _, row in autos.iterrows():
                sitio, fecha, hora = row["sitio"], row["fecha"], row["hora"]

                vent_row = buscar_por_rango(ventusky, sitio, fecha, hora)
                waze_row = buscar_por_rango(waze, sitio, fecha, hora)

                # Tomamos la primera coincidencia (si hay)
                vent_data = vent_row.iloc[0].to_dict() if not vent_row.empty else {}
                waze_data = waze_row.iloc[0].to_dict() if not waze_row.empty else {}

                combinado = {**row.to_dict(), **vent_data, **waze_data}
                resultados.append(combinado)

            df_final = pd.DataFrame(resultados)

            combinado_path = base / f"combinado_{subcarpeta.name}.csv"
            df_final.to_csv(combinado_path, index=False)
            print(f"✅ Archivo combinado guardado en {combinado_path}")

    print("\n✅ Cruce con tolerancia ±1 hora completado.")


# === EJECUCIÓN ===
if __name__ == "__main__":
    #generar_csvs_tracking("data/angamos/")
    cruzar_datos("data/angamos/")
