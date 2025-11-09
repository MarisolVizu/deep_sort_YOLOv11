import json
import pandas as pd
from datetime import datetime
from collections import defaultdict

# === Cargar datos ===
with open("data/waze_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# === Agrupar por timestamp ===
grupos = defaultdict(list)
for d in data:
    grupos[d["timestamp"]].append(d)

# === Crear tabla final ===
resultados = []
for timestamp, items in grupos.items():
    base = items[0]  # uno representativo
    texto = [i.get("texto_popup", "") for i in items]

    # detectar si hay alguno vacío o nulo
    sin_trafico = any(t is None or t.strip() == "" for t in texto)

    # unir todos los textos para buscar palabras clave
    texto_total = " ".join(t for t in texto if t)

    trafico_muy_intenso = int("Tráfico muy intenso" in texto_total)
    trafico_denso = int("Tráfico denso" in texto_total)
    trafico_moderado = int("Tráfico moderado" in texto_total)
    trafico_ligero = int("Tráfico ligero" in texto_total)

    # extraer hora, día y número de día
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    hora = dt.strftime("%H:%M")
    dia = dt.strftime("%A").lower()  # lunes, martes...
    dia_num = dt.weekday()

    resultados.append({
        "timestamp": timestamp,
        "hora": hora,
        "dia": dia,
        "dia_num": dia_num,
        "lugar": base.get("lugar", ""),
        "trafico_muy_intenso": trafico_muy_intenso,
        "trafico_denso": trafico_denso,
        "trafico_moderado": trafico_moderado,
        "trafico_ligero": trafico_ligero,
        "sin_trafico": int(sin_trafico)
    })

# === Convertir a DataFrame y exportar a CSV ===
df = pd.DataFrame(resultados)
df.to_csv("data/waze_limpio.json.csv", index=False, encoding="utf-8-sig")

print("✅ Archivo 'trafico_dummies.csv' creado correctamente.")
