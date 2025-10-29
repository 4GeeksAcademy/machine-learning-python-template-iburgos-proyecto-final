import pandas as pd
import joblib
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ========================================
# 1️⃣ Cargar modelo, escalador y dataset
# ========================================
df = joblib.load("src/archivos/df_train_scaled2.pkl")
kmeans = joblib.load("src/archivos/modelo_promociones.pkl")
scaler = joblib.load("src/archivos/scaler_promociones.pkl")

print("✅ Modelo y datos cargados correctamente")
print("Columnas:", df.columns.tolist())

# ========================================
# 2️⃣ Preparar los datos (igual que en el entrenamiento)
# ========================================
for col in ["venta", "beneficio", "margen_venta", "cantidad"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["venta", "beneficio", "margen_venta", "cantidad"])

X = df[["venta", "beneficio", "margen_venta", "cantidad"]]
X_scaled = scaler.transform(X)

# ========================================
# 3️⃣ Aplicar el modelo a los datos
# ========================================
df["cluster"] = kmeans.predict(X_scaled)
print(f"✅ Clusters asignados. Total de clusters: {df['cluster'].nunique()}")

# ========================================
# 4️⃣ Evaluación del modelo
# ========================================
# Inercia (qué tan compactos son los clusters)
inercia = kmeans.inertia_

# Silhouette Score (qué tan bien separados están los clusters)
silhouette = silhouette_score(X_scaled, df["cluster"])

print("\n📊 Métricas de evaluación:")
print(f" - Inercia (menor = mejor): {inercia:.2f}")
print(f" - Silhouette Score (entre -1 y 1, mayor = mejor): {silhouette:.3f}")

# ========================================
# 5️⃣ Resumen de clusters
# ========================================
cluster_summary = df.groupby("cluster")[["venta", "beneficio", "margen_venta", "cantidad"]].mean().round(2)
print("\n📋 Promedios por cluster:")
print(cluster_summary)