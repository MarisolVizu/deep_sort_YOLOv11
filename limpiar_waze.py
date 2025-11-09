import json
import pandas as pd
from datetime import datetime
from collections import defaultdict

def procesar_waze(input_path: str, output_path: str):
    """
    Procesa datos de tráfico de Waze agrupando por timestamp.
    Genera variables dummies según nivel de tráfico, día numérico y guarda el resultado en CSV.
    Incluye 'id' único por timestamp (sin repeticiones).
    """

    # === 1. Cargar datos ===
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # === 2. Agrupar por timestamp ===
    grupos = defaultdict(list)
    for d in data:
        grupos[d["timestamp"]].append(d)

    # === 3. Crear tabla final ===
    resultados = []
    for timestamp, items in grupos.items():
        base = items[0]  # registro representativo
        texto = [i.get("texto_popup", "") for i in items]

        # detectar si hay alguno vacío o nulo
        sin_trafico = any(t is None or t.strip() == "" for t in texto)

        # unir todos los textos para buscar palabras clave
        texto_total = " ".join(t for t in texto if t)

        trafico_muy_intenso = int("Tráfico muy intenso" in texto_total)
        trafico_denso = int("Tráfico denso" in texto_total)
        trafico_moderado = int("Tráfico moderado" in texto_total)
        trafico_ligero = int("Tráfico ligero" in texto_total)

        # === extraer hora, día y número de día ===
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        hora = dt.strftime("%H:%M")
        dia_nombre = dt.strftime("%A").lower()

        # diccionario de días
        dias_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
            "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
            "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6
        }
        dia_num = dias_map.get(dia_nombre, -1)

        # === generar id único ===
        lugar = base.get("lugar", "").strip().lower().replace(" ", "_")
        fecha_str = dt.strftime("%Y-%m-%d_%H-%M")
        id_unico = f"{lugar}_{fecha_str}"

        resultados.append({
            "id": id_unico,
            "timestamp": timestamp,
            "hora": hora,
            "dia": dia_nombre,
            "dia_num": dia_num,
            "lugar": base.get("lugar", ""),
            "trafico_muy_intenso": trafico_muy_intenso,
            "trafico_denso": trafico_denso,
            "trafico_moderado": trafico_moderado,
            "trafico_ligero": trafico_ligero,
            "sin_trafico": int(sin_trafico)
        })

    # === 4. Convertir a DataFrame y exportar ===
    df = pd.DataFrame(resultados)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ Archivo '{output_path}' creado correctamente con {len(df)} registros únicos.")
