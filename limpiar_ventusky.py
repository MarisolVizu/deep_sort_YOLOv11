import json
import csv
from datetime import datetime
from itertools import groupby
import re

def filtrar_ventusky(input_path: str, output_path: str):
    """
    Filtra registros de Ventusky agrupando por ID, deja uno por ID y limpia los campos numéricos.
    Exporta un CSV similar al formato de Waze, con hora tomada del timestamp (sin segundos).
    """

    # === 1. Cargar la data ===
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # === 2. Diccionario de días ===
    dias_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
        "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
        "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6
    }

    # === 3. Ordenar por id ===
    data.sort(key=lambda x: x["id"])

    filtrados = []

    for id_val, grupo in groupby(data, key=lambda x: x["id"]):
        grupo = list(grupo)
        mejor = None

        # === Buscar el mejor registro ===
        for g in grupo:
            try:
                ts = datetime.fromisoformat(g["timestamp"])
                dia_real = ts.weekday()
                dia_nombre = g.get("dia", "").lower()
                if dias_map.get(dia_nombre, -1) == dia_real:
                    mejor = g
                    break  # si coincide el día, tomamos este
            except Exception:
                continue

        # Si no se encontró coincidencia, tomar el primero
        if mejor is None:
            mejor = grupo[0]
            try:
                ts = datetime.fromisoformat(mejor["timestamp"])
            except Exception:
                continue

        # === extraer componentes ===
        fecha = ts.strftime("%Y-%m-%d")
        hora_str = ts.strftime("%H:%M")  # hora exacta del timestamp sin segundos

        # === limpiar valores numéricos ===
        def extraer_num(texto):
            if texto is None:
                return ""
            match = re.search(r"[-+]?\d*\.?\d+", str(texto))
            return match.group(0) if match else ""

        temperatura = extraer_num(mejor.get("temperatura", ""))
        humedad = extraer_num(mejor.get("humedad", ""))
        viento = extraer_num(mejor.get("viento", ""))
        lluvia = extraer_num(mejor.get("lluvia", ""))

        dia_nombre = mejor.get("dia", "").lower()
        dia_num = dias_map.get(dia_nombre, ts.weekday())

        # === Usar el ID original ===
        filtrados.append({
            "id": mejor["id"],
            "timestamp": mejor["timestamp"],
            "dia": dia_nombre,
            "dia_num": dia_num,
            "hora": hora_str,
            "temperatura": temperatura,
            "humedad": humedad,
            "viento": viento,
            "lluvia": lluvia,
            "fecha": fecha
        })

    # === 5. Exportar CSV ===
    columnas = ["id", "timestamp", "dia", "dia_num", "hora", "temperatura", "humedad", "viento", "lluvia", "fecha"]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(filtrados)

    print(f"✅ {len(filtrados)} registros procesados correctamente en {output_path}")
