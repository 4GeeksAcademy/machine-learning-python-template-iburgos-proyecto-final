import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# ============================
# 1) Cargar el dataset
# ============================
df = joblib.load("src/archivos/df_train_scaled2.pkl")

# Aseguramos que los campos clave sean numéricos
for col in ["venta", "beneficio", "margen_venta", "cantidad"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["venta", "beneficio", "margen_venta", "cantidad"])

# ============================
# 2) Seleccionar variables para clustering
# ============================
X = df[["venta", "beneficio", "margen_venta", "cantidad"]]

# Normalizamos los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# 3) Entrenar modelo K-Means
# ============================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Guardamos los objetos
joblib.dump(kmeans, "modelo_promociones.pkl")
joblib.dump(scaler, "scaler_promociones.pkl")

# ============================
# 4) Interpretar los grupos
# ============================
cluster_summary = df.groupby("cluster")[["venta", "beneficio", "margen_venta", "cantidad"]].mean().round(2)
print("Resumen de los clusters:")
print(cluster_summary)

# ============================
# 5) Etiquetas según perfil
# ============================
def etiquetar_cluster(row):
    if row["cluster"] == cluster_summary["venta"].idxmax():
        return "Producto Estrella 🟢"
    elif row["cluster"] == cluster_summary["venta"].idxmin():
        return "Candidato a Promoción 🔴"
    else:
        return "Estable 🟡"

df["categoria_promocion"] = df.apply(etiquetar_cluster, axis=1)

# ============================
# 6) Guardar resultado final
# ============================
df.to_csv("productos_promocion.csv", index=False)
print("✅ Archivo 'productos_promocion.csv' generado con recomendaciones.")
