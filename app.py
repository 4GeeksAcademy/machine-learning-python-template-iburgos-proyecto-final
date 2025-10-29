import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import time
from datetime import datetime

# ==============================
# 🎨 CONFIGURACIÓN GENERAL
# ==============================
st.set_page_config(page_title="SUMA Inteligencia Comercial", page_icon="🛒", layout="wide")

# --- Paleta de colores corporativa ---
COLOR_PRIMARY = "#CC0000"
COLOR_SECONDARY = "#F5F5F5"
COLOR_TEXT = "#333333"
COLOR_SUCCESS = "#00B050"
COLOR_WARNING = "#FFD700"
COLOR_DANGER = "#CC0000"

# --- Estilos globales ---
st.markdown(f"""
<style>
    .main {{
        background-color: {COLOR_SECONDARY};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {COLOR_TEXT};
        font-family: 'Helvetica', sans-serif;
    }}
    .metric-container {{
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }}
    .metric-title {{
        font-weight: bold;
        color: {COLOR_PRIMARY};
    }}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🚀 ANIMACIÓN DE CARGA
# ==============================
with st.spinner("🚀 Cargando el modelo y los datos del sistema SUMA..."):
    time.sleep(2)

# --- Encabezado con logo ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("logo.jpg", width=130)
with col_title:
    st.title("🛒 SUMA Supermercados - Plataforma de Inteligencia Comercial")

# --- Tabs principales ---
tab1, tab2, tab3, tab4 = st.tabs(["🤖 Recomendador", "💹 Dashboard de Ventas", "💡 Promociones Inteligentes", "📦 Inventario Inteligente"])


# ==============================
# 📦 CARGA DE DATOS Y MODELOS
# ==============================
@st.cache_resource
def load_model():
    knn_model = joblib.load("modelo_knn.pkl")
    product_to_idx = joblib.load("product_to_idx.pkl")
    idx_to_product = joblib.load("idx_to_product.pkl")
    sparse_matrix = joblib.load("sparse_matrix.pkl")
    return knn_model, product_to_idx, idx_to_product, sparse_matrix

knn_model, product_to_idx, idx_to_product, sparse_matrix = load_model()

@st.cache_data
def load_sales():
    df = pd.read_csv("ventas_anuales1.csv")
    for col in ["cantidad", "coste", "venta", "beneficio", "margen_coste", "margen_venta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df

df_sales = load_sales()

# ==============================
# 🤖 TAB 1 - RECOMENDADOR
# ==============================
with tab1:
    st.header("🤖 Recomendador de Productos")
    st.write("Selecciona un producto para obtener sugerencias automáticas basadas en compras similares.")

    producto = st.selectbox("🛍️ Producto:", sorted(product_to_idx.keys()))

    def recomendar_producto(producto, top=5):
        if producto not in product_to_idx:
            return []
        idx = product_to_idx[producto]
        vector = sparse_matrix.T[idx]
        distances, indices = knn_model.kneighbors(vector, n_neighbors=top + 1)
        vecinos = [idx_to_product[i] for i in indices.flatten() if idx_to_product[i] != producto]
        return vecinos[:top]

    if st.button("🔮 Mostrar Recomendaciones"):
        recs = recomendar_producto(producto)
        if recs:
            st.success(f"Basado en clientes que compraron **{producto}**, también adquirieron:")
            for r in recs:
                st.markdown(f"<div class='metric-container'>✅ {r}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ No se encontraron recomendaciones.")

# ==============================
# 💹 TAB 2 - DASHBOARD DE VENTAS
# ==============================
with tab2:
    st.header("💹 Dashboard de Ventas")
    df_sales.columns = df_sales.columns.str.lower()

    col1, col2, col3 = st.columns(3)
    total_ventas = df_sales["venta"].sum()
    total_beneficio = df_sales["beneficio"].sum()
    margen_promedio = df_sales["margen_venta"].mean()

    col1.metric("💰 Ingresos Totales", f"{total_ventas:,.0f} €")
    col2.metric("📦 Beneficio Total", f"{total_beneficio:,.0f} €")
    col3.metric("📈 Margen Promedio", f"{margen_promedio:.2f}%")

    familias = df_sales["familia_legible"].dropna().unique()
    familia_sel = st.selectbox("Filtrar por categoría:", ["Todas"] + list(familias))

    if familia_sel != "Todas":
        df_filt = df_sales[df_sales["familia_legible"] == familia_sel]
    else:
        df_filt = df_sales

    col4, col5 = st.columns(2)
    top10 = df_filt.groupby("producto")["cantidad"].sum().sort_values(ascending=False).head(10).reset_index()
    fig1 = px.bar(top10, x="producto", y="cantidad", color="cantidad", color_continuous_scale="Reds", title="📦 Top 10 más vendidos")
    col4.plotly_chart(fig1, use_container_width=True)

    top_rent = df_filt.groupby("producto")["beneficio"].sum().sort_values(ascending=False).head(10).reset_index()
    fig2 = px.bar(top_rent, x="producto", y="beneficio", color="beneficio", color_continuous_scale="Blues", title="💰 Top 10 más rentables")
    col5.plotly_chart(fig2, use_container_width=True)

    col6, col7 = st.columns(2)
    low_rent = df_filt.groupby("producto")["beneficio"].sum().sort_values().head(10).reset_index()
    fig3 = px.bar(low_rent, x="producto", y="beneficio", color="beneficio", color_continuous_scale="Greys", title="📉 Top 10 menos rentables")
    col6.plotly_chart(fig3, use_container_width=True)

    top_margin = df_filt.groupby("producto")["margen_venta"].mean().sort_values(ascending=False).head(10).reset_index()
    fig4 = px.bar(top_margin, x="producto", y="margen_venta", color="margen_venta", color_continuous_scale="OrRd", title="📊 Mayor margen (%)")
    col7.plotly_chart(fig4, use_container_width=True)

    if "fecha" in df_sales.columns:
        ventas_mes = df_sales.groupby(df_sales["fecha"].dt.to_period("M"))["venta"].sum().reset_index()
        ventas_mes["fecha"] = ventas_mes["fecha"].astype(str)
        fig5 = px.line(ventas_mes, x="fecha", y="venta", markers=True, color_discrete_sequence=[COLOR_PRIMARY], title="📅 Evolución mensual de ventas")
        st.plotly_chart(fig5, use_container_width=True)

# ==============================
# 💡 TAB 3 - PROMOCIONES
# ==============================
with tab3:
    st.header("💡 Promociones Inteligentes")

    try:
        df_promos = pd.read_csv("src/archivos/productos_promocion.csv")

        col1, col2, col3 = st.columns(3)
        total = len(df_promos)
        estrellas = (df_promos["categoria_promocion"] == "Producto Estrella 🟢").sum()
        candidatos = (df_promos["categoria_promocion"] == "Candidato a Promoción 🔴").sum()
        estables = (df_promos["categoria_promocion"] == "Estable 🟡").sum()

        col1.metric("Productos Totales", f"{total}")
        col2.metric("⭐ Estrellas", f"{estrellas}")
        col3.metric("📉 Promoción", f"{candidatos}")

        categoria_sel = st.selectbox(
            "Selecciona tipo de producto:",
            ["Todos", "Producto Estrella 🟢", "Estable 🟡", "Candidato a Promoción 🔴"]
        )

        if categoria_sel != "Todos":
            df_filt = df_promos[df_promos["categoria_promocion"] == categoria_sel]
        else:
            df_filt = df_promos

        fig = px.pie(df_promos, names="categoria_promocion", title="Distribución de Categorías",
                     color_discrete_map={
                         "Producto Estrella 🟢": "#00CC44",
                         "Estable 🟡": "#FFD700",
                         "Candidato a Promoción 🔴": "#CC0000"
                     })
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Detalle de Productos")
        st.dataframe(
            df_filt[["producto", "familia_legible", "venta", "beneficio", "margen_venta", "categoria_promocion"]]
            .sort_values(by="venta", ascending=False)
            .reset_index(drop=True)
        )

        st.info("""
        🔎 **Interpretación:**
        - 🟢 **Producto Estrella:** Alta venta y margen. Mantener visibilidad y stock.
        - 🟡 **Estable:** Buen rendimiento, pero no sobresaliente.
        - 🔴 **Candidato a Promoción:** Baja rotación o margen. Ideal para descuentos o combos.
        """)

    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'productos_promocion.csv'. Ejecuta primero el modelo de promociones.")

# ==============================
# 📦 TAB 4 - INVENTARIO INTELIGENTE
# ==============================
with tab4:
    st.header("📦 Inventario Inteligente (Predicción de Ventas)")
    st.markdown("Usamos **Facebook Prophet** para anticipar la demanda futura de los productos.")

    import os
    import joblib
    from prophet.plot import plot_plotly

    model_dir = "modelos_prophet"

    if not os.path.exists(model_dir) or len(os.listdir(model_dir)) == 0:
        st.warning("⚠️ No hay modelos entrenados. Ejecuta primero `train_prophet_inventario.py`.")
    else:
        modelos = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
        nombres = [m.replace(".pkl", "").replace("_", " ") for m in modelos]

        producto_sel = st.selectbox("🛍️ Selecciona un producto:", sorted(nombres))
        if producto_sel:
            model_path = os.path.join(model_dir, producto_sel.replace(" ", "_") + ".pkl")
            model = joblib.load(model_path)
            periodo = st.slider("Selecciona horizonte de predicción (días):", 7, 90, 30)

            # --- Generar predicción ---
            future = model.make_future_dataframe(periods=periodo)
            forecast = model.predict(future)

            # Guardamos forecast en variable general (antes se llamaba df_pred)
            df_pred = forecast

            st.subheader(f"📈 Predicción de ventas para `{producto_sel}`")
            st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)

            st.dataframe(
                forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periodo).rename(columns={
                    "ds": "Fecha",
                    "yhat": "Predicción (unidades)",
                    "yhat_lower": "Límite Inferior",
                    "yhat_upper": "Límite Superior"
                })
            )

            # ==========================
    # 📦 Recomendación de reposición
    # ==========================
    st.subheader("📦 Recomendación de Reposición Automática")

    # Stock actual ingresado por el usuario
    stock_actual = st.number_input(
        "Introduce el stock actual del producto seleccionado:",
        min_value=0, value=0, step=1
    )

    # Estimamos ventas futuras 
    ventas_previstas = df_pred["yhat"].tail(periodo).sum()
    unidades_recomendadas = max(0, round(ventas_previstas - stock_actual))

    # Mostrar resultados con estilo
    st.markdown(f"""
    🧮 **Predicción de ventas {periodo} días:** {round(ventas_previstas):,} unidades  
    📦 **Stock actual:** {stock_actual:,} unidades  
    🚚 **Recomendación:** Debes reponer **{unidades_recomendadas:,} unidades** para cubrir la demanda prevista.
    """)

    # Alerta visual
    if unidades_recomendadas > 0:
        st.error("🔴 Stock insuficiente — Se recomienda realizar un pedido de reposición.")
    else:
        st.success("🟢 Stock suficiente para cubrir la demanda prevista.")
