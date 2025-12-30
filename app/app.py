import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN DE STREAMLIT ====================
st.set_page_config(
    page_title="📊 Simulador de Ventas Noviembre 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#f093fb',
    'success': '#4CAF50',
    'warning': '#FF9800'
}

# ==================== FUNCIONES DE CARGA ROBUSTAS ====================

def cargar_modelo():
    """Carga el modelo buscando en múltiples ubicaciones"""
    posibles_rutas = [
        'app/modelo_final.joblib',
        'modelo_final.joblib',
        'models/modelo_final.joblib',
        '/mount/src/forecasting_data_science_ia/app/modelo_final.joblib'
    ]
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            try:
                return joblib.load(ruta)
            except:
                continue
    return None

def cargar_datos():
    """Carga los datos buscando en múltiples ubicaciones"""
    nombre_archivo = 'inferencia_df_transformado.csv'
    posibles_rutas = [
        f'data/processed/{nombre_archivo}',
        f'app/data/processed/{nombre_archivo}',
        nombre_archivo,
        f'../data/processed/{nombre_archivo}'
    ]
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            try:
                df = pd.read_csv(ruta)
                if 'fecha' in df.columns:
                    df['fecha'] = pd.to_datetime(df['fecha'])
                return df
            except:
                continue
    return None

# ==================== CARGA INICIAL ====================

modelo = cargar_modelo()
df = cargar_datos()

# Verificación crítica
if modelo is None:
    st.error("❌ No se pudo cargar el modelo (.joblib).")
    st.info("Asegúrate de que el archivo esté en la carpeta 'app' o 'models' en GitHub.")
    st.stop()

if df is None:
    st.error("❌ No se pudieron cargar los datos (.csv).")
    st.info("Buscando: inferencia_df_transformado.csv")
    st.stop()

# ==================== FUNCIONES DE PREDICCIÓN ====================

def hacer_predicciones_recursivas(df_producto, modelo):
    df_producto = df_producto.sort_values('fecha').reset_index(drop=True)
    columnas_modelo = modelo.feature_names_in_
    df_pred = df_producto.copy()
    
    for col_base in ['Amazon', 'Decathlon', 'Deporvillage']:
        if col_base not in df_pred.columns:
            col_x, col_y = f'{col_base}_x', f'{col_base}_y'
            if col_x in df_pred.columns and col_y in df_pred.columns:
                df_pred[col_base] = df_pred[[col_x, col_y]].mean(axis=1)
            elif col_x in df_pred.columns:
                df_pred[col_base] = df_pred[col_x]
            elif col_y in df_pred.columns:
                df_pred[col_base] = df_pred[col_y]

    predicciones = []
    for idx in range(len(df_pred)):
        try:
            X_pred = df_pred.iloc[[idx]][columnas_modelo].values
            pred = modelo.predict(X_pred)[0]
            predicciones.append(pred)
            
            if idx < len(df_pred) - 1:
                # Actualizar lags
                for col in df_pred.columns:
                    if 'lag1' in col.lower() and 'unidades' in col.lower():
                        df_pred.loc[idx + 1, col] = pred
        except Exception as e:
            st.error(f"Error en predicción: {e}")
            return None, None
            
    return np.array(predicciones), df_pred

def realizar_simulacion(df_base, producto_seleccionado, descuento_pct, escenario_competencia, modelo):
    df_sim = df_base[df_base['nombre'] == producto_seleccionado].copy()
    df_sim = df_sim.sort_values('fecha').reset_index(drop=True)
    
    if len(df_sim) == 0: return None, None
    
    precio_base = df_sim['precio_base'].iloc[0]
    df_sim['precio_venta'] = precio_base * (1 + descuento_pct / 100)
    
    factor_comp = {"Actual (0%)": 1.0, "Competencia -5%": 0.95, "Competencia +5%": 1.05}[escenario_competencia]
    
    cols_comp = ['Amazon_x', 'Decathlon_x', 'Deporvillage_x', 'Amazon_y', 'Decathlon_y', 'Deporvillage_y']
    for col in cols_comp:
        if col in df_sim.columns:
            df_sim[col] = df_sim[col] * factor_comp

    predicciones, df_final = hacer_predicciones_recursivas(df_sim, modelo)
    if predicciones is not None:
        df_final['ingresos_predicho'] = predicciones * df_final['precio_venta']
        
    return predicciones, df_final

# ==================== INTERFAZ PRINCIPAL ====================
st.markdown("""<style>
    .main-header {text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; color: white;}
    .section-divider {margin: 20px 0; border-top: 2px solid #667eea;}
</style>""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.header("🎮 Controles")
producto_seleccionado = st.sidebar.selectbox("📦 Producto:", sorted(df['nombre'].unique()))
descuento_pct = st.sidebar.slider("💰 Descuento (%)", -50, 50, 0, 5)
escenario_competencia = st.sidebar.radio("🏆 Competencia:", ["Actual (0%)", "Competencia -5%", "Competencia +5%"])

if st.sidebar.button("🚀 Simular", use_container_width=True, type="primary"):
    st.session_state.simular = True

if st.session_state.get('simular'):
    st.markdown(f'<div class="main-header"><h1>📊 Dashboard: {producto_seleccionado}</h1></div>', unsafe_allow_html=True)
    
    preds, df_res = realizar_simulacion(df, producto_seleccionado, descuento_pct, escenario_competencia, modelo)
    
    if preds is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("📦 Unidades", f"{int(preds.sum()):,}")
        c2.metric("💵 Ingresos", f"€{df_res['ingresos_predicho'].sum():,.2f}")
        c3.metric("🏷️ Descuento", f"{descuento_pct}%")
        
        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_res['dia_mes'], preds, color=COLORS['primary'], marker='o')
        ax.set_title("Ventas Diarias Noviembre")
        st.pyplot(fig)
else:
    st.info("Selecciona los parámetros y pulsa 'Simular' en el menú de la izquierda.")
