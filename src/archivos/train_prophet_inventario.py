import pandas as pd
import os
from prophet import Prophet
import joblib
from tqdm import tqdm

# ==============================
# 🔧 CONFIGURACIÓN
# ==============================
DATA_PATH = "src/archivos/tickets_trimestre_detalle.csv"
OUTPUT_DIR = "modelos_prophet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 📥 CARGAR Y PREPARAR DATOS
# ==============================
print("📂 Cargando datos...")
df = pd.read_csv(DATA_PATH)

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower()

# Asegurar que exista la columna fecha y producto
if "fecha" not in df.columns or "descripcion" not in df.columns:
    raise ValueError("El archivo debe contener las columnas 'fecha' y 'descripcion' (producto).")

# Convertir fecha a datetime
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["fecha"])

# Agrupar por producto y fecha
ventas = df.groupby(["descripcion", "fecha"])["total"].sum().reset_index()
ventas = ventas.rename(columns={"fecha": "ds", "total": "y"})

# ==============================
# 🤖 ENTRENAMIENTO PROPHET
# ==============================
productos = ventas["descripcion"].unique()
print(f"🧠 Entrenando modelos Prophet para {len(productos)} productos...")

for producto in tqdm(productos):
    df_prod = ventas[ventas["descripcion"] == producto].sort_values("ds")

    # Solo entrenar si hay datos suficientes
    if len(df_prod) < 10:
        continue

    try:
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.fit(df_prod)

        # Guardar modelo
        nombre_archivo = producto.replace(" ", "_").replace("/", "_") + ".pkl"
        ruta_modelo = os.path.join(OUTPUT_DIR, nombre_archivo)
        joblib.dump(model, ruta_modelo)
    except Exception as e:
        print(f"⚠️ Error entrenando {producto}: {e}")

print("✅ Entrenamiento completado. Modelos guardados en:", OUTPUT_DIR)
