import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN DE STREAMLIT ====================
st.set_page_config(
    page_title="📊 Simulador de Ventas Noviembre 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores original
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#f093fb',
    'success': '#4CAF50',
    'warning': '#FF9800'
}

# ==================== FUNCIONES DE CARGA ADAPTADAS A LA NUBE ====================
def cargar_modelo():
    """Carga el modelo buscando en rutas relativas de la nube"""
    # Intentamos las rutas más probables según tu estructura en GitHub
    rutas = [
        os.path.join(os.path.dirname(__file__), 'modelo_final.joblib'), # Misma carpeta que app.py
        'app/modelo_final.joblib',
        'models/modelo_final.joblib'
    ]
    for ruta in rutas:
        if os.path.exists(ruta):
            try:
                return joblib.load(ruta)
            except:
                continue
    return None

def cargar_datos():
    """Carga los datos buscando en la carpeta data/processed del repo"""
    nombre = 'inferencia_df_transformado.csv'
    rutas = [
        os.path.join('data', 'processed', nombre),
        os.path.join(os.path.dirname(__file__), 'data', 'processed', nombre),
        nombre
    ]
    for r in rutas:
        if os.path.exists(r):
            try:
                df = pd.read_csv(r)
                df['fecha'] = pd.to_datetime(df['fecha'])
                return df
            except:
                continue
    return None

# ==================== FUNCIONES DE PREDICCIÓN (TU LÓGICA ORIGINAL) ====================
def hacer_predicciones_recursivas(df_producto, modelo):
    df_producto = df_producto.sort_values('fecha').reset_index(drop=True)
    columnas_modelo = modelo.feature_names_in_
    df_pred = df_producto.copy()
    
    # Lógica de columnas de competencia
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
    # Usamos st.empty para el spinner para evitar bloqueos visuales
    with st.spinner('🔄 Calculando predicciones...'):
        for idx in range(len(df_pred)):
            try:
                X_pred = df_pred.iloc[[idx]][columnas_modelo].values
                pred = modelo.predict(X_pred)[0]
                predicciones.append(pred)
                
                if idx < len(df_pred) - 1:
                    # Actualizar lag_1
                    for col in df_pred.columns:
                        if 'lag1' in col.lower() and 'unidades' in col.lower():
                            df_pred.loc[idx + 1, col] = pred
                    # Desplazar lags
                    for lag in range(2, 8):
                        for col in df_pred.columns:
                            if f'lag{lag}' in col.lower() and 'unidades' in col.lower():
                                for prev_col in df_pred.columns:
                                    if f'lag{lag-1}' in prev_col.lower() and 'unidades' in prev_col.lower():
                                        df_pred.loc[idx + 1, col] = df_pred.loc[idx, prev_col]
                                        break
                    # Media móvil
                    for col in df_pred.columns:
                        if 'mm7' in col.lower() or 'ma7' in col.lower():
                            ultimas_preds = predicciones[-7:]
                            df_pred.loc[idx + 1, col] = np.mean(ultimas_preds)
            except:
                return None, None
    return np.array(predicciones), df_pred

def realizar_simulacion(df_base, producto_seleccionado, descuento_pct, escenario_competencia, modelo):
    df_sim = df_base[df_base['nombre'] == producto_seleccionado].copy()
    df_sim = df_sim.sort_values('fecha').reset_index(drop=True)
    if len(df_sim) == 0: return None, None
    
    precio_base = df_sim['precio_base'].iloc[0]
    df_sim['precio_venta'] = precio_base * (1 + descuento_pct / 100)
    factor_competencia = {"Actual (0%)": 1.0, "Competencia -5%": 0.95, "Competencia +5%": 1.05}[escenario_competencia]
    
    for col in ['Amazon_x', 'Decathlon_x', 'Deporvillage_x', 'Amazon_y', 'Decathlon_y', 'Deporvillage_y']:
        if col in df_sim.columns: df_sim[col] = df_sim[col] * factor_competencia
    
    # Asegurar columnas para el modelo
    for col_base in ['Amazon', 'Decathlon', 'Deporvillage']:
        col_x, col_y = f'{col_base}_x', f'{col_base}_y'
        if col_x in df_sim.columns and col_y in df_sim.columns:
            df_sim[col_base] = df_sim[[col_x, col_y]].mean(axis=1)
        elif col_x in df_sim.columns: df_sim[col_base] = df_sim[col_x]
    
    df_sim['descuento_porcentaje'] = ((df_sim['precio_venta'] - df_sim['precio_base']) / df_sim['precio_base']) * 100
    
    predicciones, df_sim_actualizado = hacer_predicciones_recursivas(df_sim, modelo)
    if predicciones is not None:
        df_sim_actualizado['ingresos_predicho'] = predicciones * df_sim_actualizado['precio_venta']
    return predicciones, df_sim_actualizado

# ==================== ESTILOS CSS ====================
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px; border-radius: 10px; color: white; margin-bottom: 20px;
    }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 4px solid #667eea; }
    .section-divider { margin-top: 30px; margin-bottom: 20px; border-top: 2px solid #667eea; }
    </style>
""", unsafe_allow_html=True)

# Carga de recursos
modelo = cargar_modelo()
df = cargar_datos()

if modelo is None:
    st.error("❌ No se encontró el modelo. Asegúrate de que 'modelo_final.joblib' esté en la carpeta 'app/' en GitHub.")
    st.stop()
if df is None:
    st.error("❌ No se encontró el CSV. Asegúrate de que esté en 'data/processed/' en GitHub.")
    st.stop()

# ==================== SIDEBAR Y LÓGICA ====================
st.sidebar.markdown("# 🎮 Controles de Simulación")
producto_seleccionado = st.sidebar.selectbox("📦 Producto:", sorted(df['nombre'].unique()))
descuento_pct = st.sidebar.slider("💰 Ajuste descuento (%)", -50, 50, 0, 5)
escenario_competencia = st.sidebar.radio("🏆 Competencia:", ["Actual (0%)", "Competencia -5%", "Competencia +5%"])

if st.sidebar.button("🚀 Simular Ventas", use_container_width=True, type="primary"):
    st.session_state.simular = True

if st.session_state.get('simular'):
    st.markdown(f'<div class="main-header"><h1>📊 Dashboard: {producto_seleccionado}</h1></div>', unsafe_allow_html=True)
    
    predicciones, df_resultados = realizar_simulacion(df, producto_seleccionado, descuento_pct, escenario_competencia, modelo)
    
    if predicciones is not None:
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 Unidades", f"{int(predicciones.sum()):,}")
        c2.metric("💵 Ingresos", f"€{df_resultados['ingresos_predicho'].sum():,.2f}")
        c3.metric("💲 Precio Prom.", f"€{df_resultados['precio_venta'].mean():.2f}")
        c4.metric("🏷️ Descuento", f"{df_resultados['descuento_porcentaje'].mean():.2f}%")
        
        # Gráfico
        st.markdown("## 📉 Predicción Diaria")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.set_style("whitegrid")
        ax.plot(df_resultados['dia_mes'], predicciones, color=COLORS['primary'], marker='o', linewidth=2)
        # Marcar Black Friday
        if 28 in df_resultados['dia_mes'].values:
            ax.axvline(28, color='red', linestyle='--', alpha=0.5)
            ax.text(28, max(predicciones), 'Black Friday', color='red', ha='center')
        st.pyplot(fig)
        
        # Tabla
        with st.expander("📋 Ver detalle diario"):
            st.dataframe(df_resultados[['fecha', 'precio_venta', 'ingresos_predicho']].tail(10))
else:
    st.info("Configura los controles y pulsa 'Simular Ventas'.")
