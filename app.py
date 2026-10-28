# -*- coding: utf-8 -*-
"""
App interactiva - ACA N.° 1 Aprendizaje Automático (CUN)
Proyecto 3 (Ventas): predicción de ventas de alto valor con 5 métodos clásicos de Machine Learning.

Cómo ejecutarla en tu computador:
    pip install -r requirements.txt
    streamlit run app.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ----------------------------------------------------------------------------------
# PASO 1: rutas y clases propias
# Los modelos K-Means y de Refuerzo son clases escritas por nosotros (clases_ml.py);
# hay que importarlas para que joblib pueda reconstruir los modelos guardados.
# ----------------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import clases_ml  # noqa: E402,F401

st.set_page_config(page_title="Predicción de ventas altas | ACA 1", page_icon="🛒", layout="wide")

# Variables que usan los modelos (las mismas del notebook, sección 1.9)
FEATURES = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category", "Product ID",
            "order_month", "order_year", "order_dayofweek", "ship_days"]
FEATS_RL = ["Product ID", "Sub-Category"]            # lo único que observa el agente de refuerzo
COLUMNAS_LOTE = ["Order Date", "Ship Date", "Ship Mode", "Segment", "Region", "Category", "Sub-Category", "Product ID"]
SIN_PRODUCTO = "(Producto nuevo / sin historial)"


# ----------------------------------------------------------------------------------
# PASO 2: cargar modelos y datos de apoyo (se guardan en memoria para ir más rápido)
# ----------------------------------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    return joblib.load(os.path.join(BASE, "modelos", "modelos_5_metodos.joblib"))


@st.cache_data
def cargar_meta():
    with open(os.path.join(BASE, "modelos", "metadatos.json"), encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_csv(nombre):
    return pd.read_csv(os.path.join(BASE, "datos", nombre))


MODELOS = cargar_modelos()
META = cargar_meta()
NOMBRES = list(MODELOS.keys())
CATALOGO = cargar_csv("catalogo_productos.csv")


# ----------------------------------------------------------------------------------
# PASO 3: funciones de predicción (idénticas a las del notebook, sección 7 y 8)
# ----------------------------------------------------------------------------------
def puntaje(nombre, X):
    """Regresión Lineal: venta estimada en USD. Los demás métodos: probabilidad de venta alta."""
    tipo, m = MODELOS[nombre]
    if tipo == "regresion_log":
        return np.clip(np.expm1(m.predict(X[FEATURES])), 0, None)
    if tipo == "agente":
        return m.predict_proba(X[FEATS_RL])[:, 1]
    return m.predict_proba(X[FEATURES])[:, 1]


def umbral_de(nombre, regla):
    clave = "umbral_decision_optimizado" if regla.startswith("Optimizado") else "umbral_natural"
    return float(META[clave][nombre])


def agregar_variables_fecha(df):
    """Crea mes, año, día de la semana y días de envío a partir de las dos fechas."""
    od, sd = pd.to_datetime(df["Order Date"]), pd.to_datetime(df["Ship Date"])
    df = df.copy()
    df["order_month"], df["order_year"] = od.dt.month, od.dt.year
    df["order_dayofweek"], df["ship_days"] = od.dt.dayofweek, (sd - od).dt.days
    return df


def leer_fechas(serie):
    """Acepta fechas día/mes/año (como train.csv) o año-mes-día."""
    try:
        return pd.to_datetime(serie, format="%d/%m/%Y")
    except (ValueError, TypeError):
        return pd.to_datetime(serie, dayfirst=True)


# ----------------------------------------------------------------------------------
# PASO 4: barra lateral (elección del método y de la regla de decisión)
# ----------------------------------------------------------------------------------
st.title("🛒 Predicción de ventas de alto valor")
st.caption("ACA N.° 1 · Aprendizaje Automático · CUN — Proyecto 3: Ventas · Dataset: Sales Forecasting (Retail Superstore)")

with st.sidebar:
    st.header("⚙️ Configuración")
    mejor = META["metodo_seleccionado"]
    metodo = st.selectbox("Método de Machine Learning", NOMBRES, index=NOMBRES.index(mejor),
                          help="Por defecto se muestra el mejor método según la evaluación del notebook.")
    regla = st.radio("Regla de decisión", ["Optimizado (recomendada)", "Natural del método"],
                     help="Optimizado: umbral que maximiza el F1 en el entrenamiento. "
                          "Natural: probabilidad ≥ 0,5 (o venta estimada ≥ umbral en USD para la Regresión Lineal).")
    st.info(f"**Venta alta** = línea de pedido de **${META['umbral_venta_usd']:,.2f} USD o más** "
            "(25 % superior del entrenamiento).")
    st.caption(f"Mejor método según el notebook: **{mejor}**")

tab_pred, tab_lote, tab_met, tab_datos, tab_info = st.tabs(
    ["🔮 Predicción", "📂 Predicción por lote", "📊 Métricas del modelo", "🔎 Exploración de datos", "ℹ️ Acerca del proyecto"])

# ----------------------------------------------------------------------------------
# PASO 5: pestaña de predicción individual
# ----------------------------------------------------------------------------------
with tab_pred:
    st.subheader("Describe una transacción y consulta la predicción")
    c1, c2, c3 = st.columns(3)
    with c1:
        categoria = st.selectbox("Categoría", sorted(CATALOGO["Category"].unique()))
        subcats = sorted(CATALOGO.loc[CATALOGO["Category"] == categoria, "Sub-Category"].unique())
        subcat = st.selectbox("Subcategoría", subcats)
        prods = CATALOGO[CATALOGO["Sub-Category"] == subcat].sort_values("Product Name")
        etiquetas = {f"{r['Product Name'][:70]}  [{r['Product ID']}]": r["Product ID"] for _, r in prods.iterrows()}
        producto_txt = st.selectbox("Producto", [SIN_PRODUCTO] + list(etiquetas.keys()),
                                    help="Si el producto no está en el catálogo, el modelo usa el promedio general.")
    with c2:
        ship_mode = st.selectbox("Tipo de envío", ["Standard Class", "Second Class", "First Class", "Same Day"])
        segmento = st.selectbox("Segmento de cliente", ["Consumer", "Corporate", "Home Office"])
        region = st.selectbox("Región", ["West", "East", "Central", "South"])
    with c3:
        f_pedido = st.date_input("Fecha del pedido", value=pd.Timestamp("2018-11-20"))
        f_envio = st.date_input("Fecha de envío", value=pd.Timestamp("2018-11-25"))

    product_id = etiquetas.get(producto_txt, "NUEVO-SIN-HISTORIAL")
    if pd.Timestamp(f_envio) < pd.Timestamp(f_pedido):
        st.error("La fecha de envío no puede ser anterior a la del pedido.")
    else:
        fila = agregar_variables_fecha(pd.DataFrame([{
            "Order Date": f_pedido, "Ship Date": f_envio, "Ship Mode": ship_mode, "Segment": segmento,
            "Region": region, "Category": categoria, "Sub-Category": subcat, "Product ID": product_id}]))
        s = float(puntaje(metodo, fila)[0])
        es_alta = s >= umbral_de(metodo, regla)
        es_usd = MODELOS[metodo][0] == "regresion_log"

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Venta estimada (USD)" if es_usd else "Probabilidad de venta alta",
                  f"${s:,.2f}" if es_usd else f"{s:.1%}")
        m2.metric("Umbral aplicado", f"${umbral_de(metodo, regla):,.2f}" if es_usd else f"{umbral_de(metodo, regla):.2f}")
        m3.metric("Resultado", "VENTA ALTA ✅" if es_alta else "Venta normal")
        if es_alta:
            st.success("Esta transacción se clasifica como **venta de alto valor**: conviene priorizarla "
                       "(inventario, atención al cliente, logística).")
        else:
            st.info("Esta transacción se clasifica como **venta normal**.")
        if producto_txt == SIN_PRODUCTO:
            st.warning("Producto sin historial: el modelo no conoce su precio y la predicción es menos fiable. "
                       "Revísala con criterio humano.")

        # Comparación entre los 5 métodos para la misma transacción
        filas = []
        for n in NOMBRES:
            sc = float(puntaje(n, fila)[0])
            usd = MODELOS[n][0] == "regresion_log"
            filas.append({"Método": n,
                          "Venta estimada (USD)": f"${sc:,.2f}" if usd else "—",
                          "Probabilidad de venta alta": "—" if usd else f"{sc:.1%}",
                          "¿Venta alta?": "SÍ" if sc >= umbral_de(n, regla) else "no"})
        st.markdown("**Los cinco métodos para esta misma transacción**")
        st.dataframe(pd.DataFrame(filas).set_index("Método"))

# ----------------------------------------------------------------------------------
# PASO 6: pestaña de predicción por lote (archivo CSV)
# ----------------------------------------------------------------------------------
with tab_lote:
    st.subheader("Sube un archivo CSV con varias transacciones")
    st.write("Columnas obligatorias: " + ", ".join(f"`{c}`" for c in COLUMNAS_LOTE) +
             ". Si incluye `Sales`, se calculan además las métricas reales.")
    with open(os.path.join(BASE, "datos", "ejemplo_lote.csv"), "rb") as f:
        st.download_button("⬇️ Descargar archivo de ejemplo", f.read(), file_name="ejemplo_lote.csv", mime="text/csv")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        datos = pd.read_csv(archivo)
        faltan = [c for c in COLUMNAS_LOTE if c not in datos.columns]
        if faltan:
            st.error(f"Faltan columnas: {faltan}")
        else:
            datos["Order Date"], datos["Ship Date"] = leer_fechas(datos["Order Date"]), leer_fechas(datos["Ship Date"])
            X = agregar_variables_fecha(datos)
            sc = puntaje(metodo, X)
            res = datos.copy()
            res["puntaje" if MODELOS[metodo][0] != "regresion_log" else "venta_estimada_usd"] = np.round(sc, 3)
            res["prediccion_venta_alta"] = (sc >= umbral_de(metodo, regla)).astype(int)
            st.success(f"{len(res)} transacciones analizadas · {int(res['prediccion_venta_alta'].sum())} marcadas como venta alta.")
            st.dataframe(res.head(200))
            st.download_button("⬇️ Descargar resultados (CSV)", res.to_csv(index=False).encode("utf-8"),
                               file_name="predicciones.csv", mime="text/csv")
            if "Sales" in datos.columns:
                from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                real = (datos["Sales"] >= META["umbral_venta_usd"]).astype(int)
                pred = res["prediccion_venta_alta"]
                a, b, c, d = st.columns(4)
                a.metric("Accuracy", f"{accuracy_score(real, pred):.3f}")
                b.metric("Precision", f"{precision_score(real, pred, zero_division=0):.3f}")
                c.metric("Recall", f"{recall_score(real, pred, zero_division=0):.3f}")
                d.metric("F1", f"{f1_score(real, pred, zero_division=0):.3f}")

# ----------------------------------------------------------------------------------
# PASO 7: pestaña de métricas (resultados de la evaluación del notebook)
# ----------------------------------------------------------------------------------
with tab_met:
    st.subheader("Desempeño de los cinco métodos en el conjunto de prueba")
    nat = pd.DataFrame(META["metricas_prueba_regla_natural"]).T.round(3)
    opt = pd.DataFrame(META["metricas_prueba_umbral_optimizado"]).T.round(3)
    st.markdown("**Regla natural de cada método**")
    st.dataframe(nat)
    st.markdown("**Umbral optimizado en el entrenamiento** (para la Regresión Lineal el umbral está en USD)")
    st.dataframe(opt)
    st.markdown("**PR-AUC en prueba** (métrica principal de selección; el azar ≈ 0,26)")
    st.bar_chart(nat["pr_auc"])
    st.markdown("**Regresión Lineal como estimador de montos (USD)**")
    st.dataframe(pd.DataFrame(META["regresion_prueba"]).T.round(3))
    st.info(f"Método seleccionado: **{META['metodo_seleccionado']}**. {META['nota_seleccion']}")

# ----------------------------------------------------------------------------------
# PASO 8: pestaña de exploración de datos
# ----------------------------------------------------------------------------------
with tab_datos:
    st.subheader("Lo que muestran los datos históricos (2015-2018)")
    sub = cargar_csv("resumen_subcategoria.csv").set_index("Sub-Category")
    a, b = st.columns(2)
    a.markdown("**Mediana de la venta por subcategoría (USD)**")
    a.bar_chart(sub["mediana_usd"].sort_values())
    b.markdown("**% de líneas que son venta alta, por subcategoría**")
    b.bar_chart(sub["pct_venta_alta"].sort_values())
    st.markdown("**Ventas totales por mes (USD)**")
    mensual = cargar_csv("ventas_mensuales.csv").set_index("mes")
    st.line_chart(mensual["ventas_usd"])

# ----------------------------------------------------------------------------------
# PASO 9: pestaña informativa
# ----------------------------------------------------------------------------------
with tab_info:
    st.subheader("Acerca del proyecto")
    st.markdown(f"""
**Objetivo general.** Desarrollar una solución integral de Machine Learning clásico aplicada a un problema de ventas:
desde la limpieza de datos hasta el entrenamiento de modelos predictivos y su despliegue en esta interfaz.

**Pregunta orientadora.** ¿Cómo transformar datos de ventas en conocimiento predictivo mediante técnicas clásicas de
Machine Learning y presentarlo en una interfaz interactiva accesible para usuarios finales?

**Flujo metodológico.** 1) EDA → 2) Limpieza → 3) Modelado clásico → 4) Evaluación → 5) Despliegue en Streamlit.

**Métodos:** Regresión Lineal, Regresión Logística, Árbol de Decisión, K-Means (usado como clasificador) y Aprendizaje por Refuerzo.

**Datos:** {META['n_entrenamiento']:,} líneas de pedido para entrenar y {META['n_prueba']:,} para probar (partición por pedido).

**Limitaciones.** El dataset no incluye cantidad ni descuento, por lo que el monto exacto no puede predecirse con precisión.
Para productos nuevos el modelo es menos fiable. Las predicciones apoyan la decisión humana; no la reemplazan.
""")
