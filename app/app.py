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

# ==================== CARGA DE RECURSOS (ROBUSTA) ====================

def cargar_modelo():
    """Carga el modelo desde la carpeta de la app"""
    try:
        # Intenta ruta relativa al archivo app.py
        ruta = os.path.join(os.path.dirname(__file__), 'modelo_final.joblib')
        if os.path.exists(ruta):
            with open(ruta, 'rb') as f:
                return joblib.load(f)
        
        # Ruta alternativa estándar en Streamlit Cloud
        ruta_alt = 'app/modelo_final.joblib'
        if os.path.exists(ruta_alt):
            return joblib.load(ruta_alt)
    except Exception as e:
        st.error(f"Error técnico al cargar modelo: {e}")
        return None
    return None

def cargar_datos():
    """Busca el CSV en todas las ubicaciones posibles del repo"""
    nombre = 'inferencia_df_transformado.csv'
    
    # Lista exhaustiva de rutas posibles en GitHub
    rutas = [
        os.path.join('data', 'processed', nombre),
        os.path.join('app', 'data', 'processed', nombre),
        os.path.join(os.path.dirname(__file__), 'data', 'processed', nombre),
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

# ==================== EJECUCIÓN DE CARGA ====================
modelo = cargar_modelo()
df = cargar_datos()

# Verificación de recursos
if modelo is None:
    st.error("❌ El modelo existe pero no pudo cargarse (revisa la versión de scikit-learn en requirements.txt).")
    st.stop()

if df is None:
    st.error("❌ No se encuentra el archivo CSV de datos.")
    st.info("Asegúrate de que 'inferencia_df_transformado.csv' esté en la carpeta 'data/processed' en GitHub.")
    st.stop()

# ==================== FUNCIONES DE LÓGICA ====================

def hacer_predicciones_recursivas(df_producto, modelo):
    df_producto = df_producto.sort_values('fecha').reset_index(drop=True)
    columnas_modelo = modelo.feature_names_in_
    df_pred = df_producto.copy()
    
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
            if idx < len(df_pred) - 1:
                # Actualización de Lags para el día siguiente
                for col in df_pred.columns:
                    if 'lag1' in col.lower() and 'unidades' in col.lower():
                        df_pred.loc[idx + 1, col] = pred
        except Exception as e:
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
        if c in df_sim.columns: 
            df_sim[c] = df_sim[c] * f_comp
            
    preds, df_f = hacer_predicciones_recursivas(df_sim, mod)
    if preds is not None:
        df_f['ingresos_predicho'] = preds * df_f['precio_venta']
    return preds, df_f

# ==================== INTERFAZ DE USUARIO ====================
if 'sim' not in st.session_state:
    st.session_state.sim = False

st.sidebar.header("🎮 Controles de Simulación")
producto_sel = st.sidebar.selectbox("📦 Selecciona Producto:", sorted(df['nombre'].unique()))
desc_sel = st.sidebar.slider("💰 Ajuste Precio (%)", -50, 50, 0, 5)
