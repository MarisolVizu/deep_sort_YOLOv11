import json

# === 1. Cargar el archivo JSON ===
with open("data/ventusky_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# === 2. Especificar el campo del que quieres ver los valores únicos ===
campo = "id"

# === 3. Obtener valores únicos ===
valores_unicos = {item[campo] for item in data if campo in item}

# === 4. Mostrar los resultados ===
print(f"Valores únicos en '{campo}':", valores_unicos)
