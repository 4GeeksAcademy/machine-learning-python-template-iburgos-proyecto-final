import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors


# 1️ Cargar modelo, diccionarios y dataset

knn_model = joblib.load("src/archivos/modelo_knn.pkl")
product_to_idx = joblib.load("src/archivos/product_to_idx.pkl")
idx_to_product = joblib.load("src/archivos/idx_to_product.pkl")
df = joblib.load("src/archivos/df_train1.pkl")

print("✅ Modelo y dataset cargados correctamente")
print("Columnas del dataset:", df.columns.tolist())


# 2️ Reconstruir la matriz 'basket_binary'

# Agrupamos por ticket y descripción (productos)
basket = (
    df.groupby(["ticket", "descripcion"])["cantidad"]
      .sum()
      .unstack()
      .fillna(0)
)

# Convertimos a binario (1 = comprado, 0 = no comprado)
basket_binary = basket.map(lambda x: 1 if x > 0 else 0)

print(f"✅ Matriz binaria creada con forma {basket_binary.shape}")


# 2. Función para Precision@K

def precision_at_k(model, basket_binary, product_to_idx, idx_to_product, k=5, sample_size=50):
    productos = np.random.choice(list(product_to_idx.keys()), size=sample_size, replace=False)
    precision_scores = []
    
    for prod in productos:
        idx = product_to_idx[prod]
        distancias, indices = model.kneighbors(basket_binary.values.T[idx].reshape(1, -1), n_neighbors=k+1)
        recomendados = [idx_to_product[i] for i in indices.flatten() if i != idx]

        # Productos realmente comprados junto con el producto base
        tickets_con_prod = basket_binary[basket_binary.iloc[:, idx] == 1]
        productos_reales = set(tickets_con_prod.columns[(tickets_con_prod.sum(axis=0) > 0)])

        aciertos = len(set(recomendados) & productos_reales)
        precision = aciertos / k
        precision_scores.append(precision)
    
    return np.mean(precision_scores)


# 3. Evaluar el modelo

score = precision_at_k(knn_model, basket_binary, product_to_idx, idx_to_product, k=5)
print(f"📊 Precision@5 promedio: {score:.2f}")