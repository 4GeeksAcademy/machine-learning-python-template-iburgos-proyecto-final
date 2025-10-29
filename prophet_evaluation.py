# evaluate_prophet_short_series.py
import os
import numpy as np
import pandas as pd
import joblib
from prophet import Prophet
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "src/archivos/tickets_trimestre_detalle.csv"
MODELS_DIR = "src/archivos/modelos_prophet"

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))

print("📂 Cargando datos originales...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["fecha"])

# Agrega por producto y día
ventas = (
    df.groupby(["descripcion", "fecha"])["total"]
      .sum()
      .reset_index()
      .rename(columns={"fecha":"ds","total":"y"})
)
ventas["descripcion_norm"] = ventas["descripcion"].str.lower().str.strip()

archivos_modelos = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
print(f"🧩 Encontrados {len(archivos_modelos)} modelos Prophet para evaluar...\n")

resultados = []

for archivo in tqdm(archivos_modelos):
    producto_from_file = archivo[:-4].replace("_"," ").lower().strip()
    ruta = os.path.join(MODELS_DIR, archivo)

    # Serie del producto por coincidencia flexible
    df_prod = ventas[ventas["descripcion_norm"].str.contains(producto_from_file, na=False)].sort_values("ds")
    if df_prod.empty:
        # intento exacto por si el nombre ya viene completo
        df_prod = ventas[ventas["descripcion_norm"] == producto_from_file].sort_values("ds")
    n = len(df_prod)
    if n < 21:  # mínimo ~3 semanas de datos para probar
        continue

    # Split temporal 80/20
    cutoff = int(n*0.8)
    train, test = df_prod.iloc[:cutoff].copy(), df_prod.iloc[cutoff:].copy()

    # Entrena Prophet SOLO con estacionalidad semanal (apto para 3 meses)
    try:
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative"
        )
        model.fit(train[["ds","y"]])

        # Predice exactamente las fechas del test
        future = test[["ds"]]
        fcst = model.predict(future)

        y_true = test["y"].values
        y_hat  = fcst["yhat"].values

        mae  = mean_absolute_error(y_true, y_hat)
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        mape_val = mape(y_true, y_hat)

        # Baselines
        # 1) naive: ŷ_t = y_{t-1}
        y_hat_naive = test["y"].shift(1).fillna(method="bfill").values
        mae_naive  = mean_absolute_error(y_true, y_hat_naive)
        rmse_naive = np.sqrt(mean_squared_error(y_true, y_hat_naive))
        mape_naive = mape(y_true, y_hat_naive)

        # 2) naive semanal (siempre que existan al menos 7 observaciones previas)
        # ŷ_t = y_{t-7}, si no existe usa y_{t-1}
        y_all = pd.concat([train["y"], test["y"]], ignore_index=True)
        y_hat_week = []
        for i in range(cutoff, cutoff+len(test)):
            if i-7 >= 0:
                y_hat_week.append(y_all.iloc[i-7])
            else:
                y_hat_week.append(y_all.iloc[i-1])
        y_hat_week = np.array(y_hat_week)

        mae_week  = mean_absolute_error(y_true, y_hat_week)
        rmse_week = np.sqrt(mean_squared_error(y_true, y_hat_week))
        mape_week = mape(y_true, y_hat_week)

        resultados.append({
            "producto": df_prod["descripcion"].iloc[0],
            "n_obs": n,
            "mae": mae, "rmse": rmse, "mape": mape_val,
            "mae_naive": mae_naive, "rmse_naive": rmse_naive, "mape_naive": mape_naive,
            "mae_week": mae_week, "rmse_week": rmse_week, "mape_week": mape_week
        })

    except Exception as e:
        print(f"⚠️ Error evaluando {archivo}: {e}")

if not resultados:
    print("⚠️ No se generaron resultados. Verifica que los nombres coincidan y que cada serie tenga suficientes días.")
else:
    res = pd.DataFrame(resultados)
    # Comparativa rápida frente a baselines
    res["mejora_vs_naive_%"] = 100*(res["mape_naive"] - res["mape"]) / res["mape_naive"]
    res["mejora_vs_naive_week_%"] = 100*(res["mape_week"] - res["mape"]) / res["mape_week"]

    print("\n🏆 Top 10 productos por menor MAPE (Prophet):")
    cols = ["producto","n_obs","mape","mape_naive","mape_week","mejora_vs_naive_%","mejora_vs_naive_week_%"]
    print(res.sort_values("mape").head(10)[cols].to_string(index=False))

    print("\n📉 Promedios globales:")
    print(res[["mae","rmse","mape","mape_naive","mape_week","mejora_vs_naive_%","mejora_vs_naive_week_%"]]
          .mean().round(4).to_dict())

    # (Opcional) muestra 5 peores para identificar problemas
    print("\n🐢 5 productos con peor MAPE (Prophet):")
    print(res.sort_values("mape", ascending=False).head(5)[cols].to_string(index=False))

print("\n✅ Evaluación completada.")
# Filtrar valores razonables (descarta mape exagerados)
res_filtrado = res[(res["mape"] > 0) & (res["mape"] < 2)]
print("\n📊 Métricas promedio (solo MAPE razonables):")
print(res_filtrado[["mae", "rmse", "mape"]].mean().round(4).to_dict())
