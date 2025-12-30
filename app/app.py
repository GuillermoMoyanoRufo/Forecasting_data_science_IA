import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN DE STREAMLIT ====================
st.set_page_config(
    page_title="📊 Simulador de Ventas Noviembre 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores Samsung
COLORS = {
    'primary': '#1428a0', # Azul Samsung
    'secondary': '#764ba2',
    'accent': '#f093fb',
    'success': '#4CAF50',
    'warning': '#FF9800'
}

# ==================== RASTREADOR DE ARCHIVOS (DEBUG) ====================
st.sidebar.markdown("### 🔍 Estado del Servidor")
actual_dir = os.getcwd()
st.sidebar.write(f"📁 Directorio: `{actual_dir}`")

# Función para intentar cargar el modelo
def cargar_modelo():
    import joblib
    # Intentamos las rutas más probables en Streamlit Cloud
    rutas = [
        'app/modelo_final.joblib',
        'modelo_final.joblib',
        'models/modelo_final.joblib',
        '/mount/src/forecasting_data_science_ia/app/modelo_final.joblib'
    ]
    
    for r in rutas:
        if os.path.exists(r):
            try:
                return joblib.load(r)
            except:
                continue
    return None

# Función para intentar cargar los datos
def cargar_datos():
    nombre = 'inferencia_df_transformado.csv'
    rutas = [
        f'data/processed/{nombre}',
        f'app/data/processed/{nombre}',
        nombre,
        f'app/{nombre}'
    ]
    for r in rutas:
        if os.path.exists(r):
            try:
                df = pd.read_csv(r)
                if 'fecha' in df.columns:
                    df['fecha'] = pd.to_datetime(df['fecha'])
                return df
            except:
                continue
    return None

# ==================== CARGA DE RECURSOS ====================
modelo = cargar_modelo()
df = cargar_datos()

# Verificación de seguridad
if modelo is None:
    st.error("❌ ERROR: No se encuentra 'modelo_final.joblib'")
    st.info("Sube el archivo directamente a la carpeta 'app' en GitHub.")
    # Mostrar qué archivos ve el sistema para ayudar a diagnosticar
    if os.path.exists('app'):
        st.write("Archivos en /app:", os.listdir('app'))
    else:
        st.write("Archivos en raíz:", os.listdir('.'))
    st.stop()

if df is None:
    st.error("❌ ERROR: No se encuentra el CSV de datos.")
    st.stop()

# ==================== FUNCIONES DE LÓGICA ====================
def hacer_predicciones_recursivas(df_producto, modelo):
    df_producto = df_producto.sort_values('fecha').reset_index(drop=True)
    columnas_modelo = modelo.feature_names_in_
    df_pred = df_producto.copy()
    
    # Asegurar columnas de competencia
    for col_base in ['Amazon', 'Decathlon', 'Deporvillage']:
        if col_base not in df_pred.columns:
            for suf in ['_x', '_y']:
                if f'{col_base}{suf}' in df_pred.columns:
                    df_pred[col_base] = df_pred[f'{col_base}{suf}']
                    break

    predicciones = []
    for idx in range(len(df_pred)):
        try:
            X_pred = df_pred.iloc[[idx]][columnas_modelo].values
            pred = modelo.predict(X_pred)[0]
            predicciones.append(pred)
            
            # Actualizar lags para el día siguiente
            if idx < len(df_pred) - 1:
                for col in df_pred.columns:
                    if 'lag1' in col.lower() and 'unidades' in col.lower():
                        df_pred.loc[idx + 1, col] = pred
        except:
            return None, None
            
    return np.array(predicciones), df_pred

def realizar_simulacion(df_base, producto, desc, escen, mod):
    df_sim = df_base[df_base['nombre'] == producto].copy()
    df_sim = df_sim.sort_values('fecha').reset_index(drop=True)
    
    precio_base = df_sim['precio_base'].iloc[0]
    df_sim['precio_venta'] = precio_base * (1 + desc / 100)
    
    f_comp = {"Actual (0%)": 1.0, "Competencia -5%": 0.95, "Competencia +5%": 1.05}[escen]
    cols_c = ['Amazon_x', 'Decathlon_x', 'Deporvillage_x', 'Amazon_y', 'Decathlon_y', 'Deporvillage_y']
    for c in cols_c:
        if c in df_sim.columns: df_sim[c] = df_sim[c] * f_comp

    preds, df_f = hacer_predicciones_recursivas(df_sim, mod)
    if preds is not None:
        df_f['ingresos_predicho'] = preds * df_f['precio_venta']
    return preds, df_f

# ==================== INTERFAZ ====================
st.sidebar.markdown("---")
producto_sel = st.sidebar.selectbox("📦 Producto:", sorted(df['nombre'].unique()))
desc_sel = st.sidebar.slider("💰 Ajuste Precio (%)", -50, 50, 0, 5)
escen_sel = st.sidebar.radio("🏆 Competencia:", ["Actual (0%)", "Competencia -5%", "Competencia +5%"])

if st.sidebar.button("🚀 Ejecutar Simulación", use_container_width=True, type="primary"):
    st.session_state.sim = True

if st.session_state.get('sim'):
    st.markdown(f"<h1 style='text-align: center; color: #1428a0;'>📊 Dashboard: {producto_sel}</h1>", unsafe_allow_html=True)
    
    preds, res = realizar_simulacion(df, producto_sel, desc_sel, escen_sel, modelo)
    
    if preds is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("📦 Unidades Totales", f"{int(preds.sum()):,}")
        c2.metric("💵 Ingresos Est.", f"€{res['ingresos_predicho'].sum():,.2f}")
        c3.metric("📉 Variación Precio", f"{desc_sel}%")
        
        # Gráfico simple
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(res['dia_mes'], preds, marker='o', color='#1428a0', linewidth=2)
        ax.set_title("Predicción Ventas Noviembre 2025")
        ax.set_xlabel("Día")
        ax.set_ylabel("Unidades")
        st.pyplot(fig)
else:
    st.info("Configura los parámetros a la izquierda y pulsa 'Ejecutar Simulación'.")
