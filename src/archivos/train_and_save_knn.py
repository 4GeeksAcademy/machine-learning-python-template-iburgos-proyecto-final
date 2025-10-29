import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import joblib

# ============================
# 1. Cargar dataset base
# ============================
df = joblib.load("src/archivos/df_train1.pkl")

# Comprobamos columnas importantes
print("Columnas del dataset:", df.columns.tolist())
print("Ejemplo de datos:")
print(df.head())

# ============================
# 2. Preprocesamiento
# ============================
# Agrupar por ticket y producto
basket = (
    df.groupby(["ticket", "descripcion"])["cantidad"]
      .sum()
      .unstack()
      .fillna(0)
)

# Convertir a binario (0 o 1)
basket_binary = basket.map(lambda x: 1 if x > 0 else 0)

# Crear matriz dispersa (más eficiente para KNN)
sparse_matrix = csr_matrix(basket_binary.values)

# ============================
# 3. Entrenar modelo KNN
# ============================
knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model.fit(sparse_matrix.T)  # Entrenamos por productos (columnas)

print(f"✅ Modelo entrenado con {sparse_matrix.shape[1]} productos.")

# ============================
# 4. Diccionarios auxiliares
# ============================
product_to_idx = {p: i for i, p in enumerate(basket_binary.columns)}
idx_to_product = {i: p for i, p in enumerate(basket_binary.columns)}

# ============================
# 5. Guardar archivos para Streamlit
# ============================
joblib.dump(knn_model, "modelo_knn.pkl")
joblib.dump(product_to_idx, "product_to_idx.pkl")
joblib.dump(idx_to_product, "idx_to_product.pkl")
joblib.dump(sparse_matrix, "sparse_matrix.pkl")

print("✅ Archivos guardados correctamente en la carpeta actual.")