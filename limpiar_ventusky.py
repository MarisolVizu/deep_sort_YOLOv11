import json
from datetime import datetime
from itertools import groupby
###  hay un unico registro por cada timestamp que encuentre


# === 1. Cargar la data ===
with open("data/ventusky_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# === 2. Mapeo de días en español a weekday (0=lunes ... 6=domingo) ===
dias_map = {
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
    "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6
}

# === 3. Ordenamos por timestamp (necesario para groupby) ===
data.sort(key=lambda x: x["timestamp"])

# === 4. Agrupar por timestamp y quedarnos con el más cercano ===
filtrados = []

for timestamp, grupo in groupby(data, key=lambda x: x["timestamp"]):
    grupo = list(grupo)
    ts = datetime.fromisoformat(timestamp)
    dia_real = ts.weekday()
    hora_scrap_min = ts.hour * 60 + ts.minute

    # Filtramos solo los que coinciden en día
    grupo = [g for g in grupo if dias_map.get(g["dia"].lower(), -1) == dia_real]

    if not grupo:
        continue  # si ninguno coincide con el día, lo saltamos

    # Calcular diferencia entre hora del campo y hora del timestamp
    for g in grupo:
        hora_dato = datetime.strptime(g["hora"], "%H:%M")
        hora_dato_min = hora_dato.hour * 60 + hora_dato.minute
        g["diff"] = abs(hora_dato_min - hora_scrap_min)

    # Elegir el registro con la menor diferencia
    mejor = min(grupo, key=lambda x: x["diff"])
    filtrados.append(mejor)

# === 5. Guardar los resultados filtrados ===
with open("data/ventusky_limpio.json", "w", encoding="utf-8") as f:
    json.dump(filtrados, f, indent=4, ensure_ascii=False)

print(f"✅ Registros finales: {len(filtrados)} guardados correctamente.")
