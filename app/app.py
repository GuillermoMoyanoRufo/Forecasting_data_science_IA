import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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

# Paleta de colores
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#f093fb',
    'success': '#4CAF50',
    'warning': '#FF9800'
}

# ==================== FUNCIONES DE CARGA ====================
def cargar_modelo():
    """Carga el modelo guardado de forma robusta"""
    import os
    import joblib
    
    # Intentamos varias rutas posibles para que funcione en local y en la nube
    posibles_rutas = [
        'app/modelo_final.joblib',           # Ruta estándar en Streamlit Cloud
        'modelo_final.joblib',               # Si se ejecuta desde dentro de /app
        'models/modelo_final.joblib',        # Tu ruta original
        '/mount/src/forecasting_data_science_ia/app/modelo_final.joblib' # Ruta absoluta nube
    ]
    
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            try:
                return joblib.load(ruta)
            except Exception as e:
                continue
    return None

def cargar_datos():
    """Carga los datos de inferencia preparados"""
    import os
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(script_dir)
        ruta_datos = os.path.join(workspace_dir, 'data', 'processed', 'inferencia_df_transformado.csv')
        
        if os.path.exists(ruta_datos):
            df = pd.read_csv(ruta_datos)
            df['fecha'] = pd.to_datetime(df['fecha'])
            return df
        else:
            return None
    except Exception as e:
        return None

# ==================== FUNCIONES DE PREDICCIÓN ====================
def hacer_predicciones_recursivas(df_producto, modelo):
    """
    Realiza predicciones recursivas para todo noviembre,
    actualizando los lags después de cada predicción
    """
    # Ordenar por fecha
    df_producto = df_producto.sort_values('fecha').reset_index(drop=True)
    
    # Columnas que el modelo espera
    columnas_modelo = modelo.feature_names_in_
    
    # Copiar dataframe para no modificar el original
    df_pred = df_producto.copy()
    
    # Crear columnas Amazon, Decathlon, Deporvillage si no existen
    # Combinando las versiones _x e _y o usando la que esté disponible
    for col_base in ['Amazon', 'Decathlon', 'Deporvillage']:
        if col_base not in df_pred.columns:
            col_x = f'{col_base}_x'
            col_y = f'{col_base}_y'
            
            if col_x in df_pred.columns and col_y in df_pred.columns:
                # Promediar si ambas existen
                df_pred[col_base] = df_pred[[col_x, col_y]].mean(axis=1)
            elif col_x in df_pred.columns:
                df_pred[col_base] = df_pred[col_x]
            elif col_y in df_pred.columns:
                df_pred[col_base] = df_pred[col_y]
    
    predicciones = []
    
    with st.spinner('🔄 Realizando predicciones recursivas...'):
        for idx in range(len(df_pred)):
            # Seleccionar características para esta predicción
            try:
                X_pred = df_pred.iloc[[idx]][columnas_modelo].values
            except KeyError as e:
                st.error(f"❌ Error: Faltan columnas {e}")
                return None, None
            
            # Hacer predicción
            pred = modelo.predict(X_pred)[0]
            predicciones.append(pred)
            
            # Si no es el último día, actualizar lags para el próximo día
            if idx < len(df_pred) - 1:
                # Actualizar lag_1 con la predicción actual (buscar la columna exacta)
                for col in df_pred.columns:
                    if 'lag1' in col.lower() and 'unidades' in col.lower():
                        df_pred.loc[idx + 1, col] = pred
                
                # Desplazar los demás lags
                for lag in range(2, 8):
                    for col in df_pred.columns:
                        if f'lag{lag}' in col.lower() and 'unidades' in col.lower():
                            for prev_col in df_pred.columns:
                                if f'lag{lag-1}' in prev_col.lower() and 'unidades' in prev_col.lower():
                                    df_pred.loc[idx + 1, col] = df_pred.loc[idx, prev_col]
                                    break
                
                # Actualizar media móvil con las últimas 7 predicciones
                for col in df_pred.columns:
                    if 'mm7' in col.lower() or 'ma7' in col.lower():
                        ultimas_preds = predicciones[-7:]
                        df_pred.loc[idx + 1, col] = np.mean(ultimas_preds)
    
    return np.array(predicciones), df_pred

def realizar_simulacion(df_base, producto_seleccionado, descuento_pct, escenario_competencia, modelo):
    """
    Realiza la simulación completa con los parámetros del usuario
    """
    # Filtrar producto
    df_sim = df_base[df_base['nombre'] == producto_seleccionado].copy()
    df_sim = df_sim.sort_values('fecha').reset_index(drop=True)
    
    if len(df_sim) == 0:
        st.error("❌ No hay datos para el producto seleccionado")
        return None, None
    
    # Recalcular precio_venta según descuento
    precio_base = df_sim['precio_base'].iloc[0]
    df_sim['precio_venta'] = precio_base * (1 + descuento_pct / 100)
    
    # Recalcular precio_competencia según escenario
    if escenario_competencia == "Actual (0%)":
        factor_competencia = 1.0
    elif escenario_competencia == "Competencia -5%":
        factor_competencia = 0.95
    elif escenario_competencia == "Competencia +5%":
        factor_competencia = 1.05
    
    # Ajustar precios de competencia individuales (con sufijos _x, _y)
    for col in ['Amazon_x', 'Decathlon_x', 'Deporvillage_x', 'Amazon_y', 'Decathlon_y', 'Deporvillage_y']:
        if col in df_sim.columns:
            df_sim[col] = df_sim[col] * factor_competencia
    
    # Crear columnas Amazon, Decathlon, Deporvillage para el modelo
    for col_base in ['Amazon', 'Decathlon', 'Deporvillage']:
        col_x = f'{col_base}_x'
        col_y = f'{col_base}_y'
        
        if col_x in df_sim.columns and col_y in df_sim.columns:
            df_sim[col_base] = df_sim[[col_x, col_y]].mean(axis=1)
        elif col_x in df_sim.columns:
            df_sim[col_base] = df_sim[col_x]
        elif col_y in df_sim.columns:
            df_sim[col_base] = df_sim[col_y]
    
    # Recalcular descuento_porcentaje
    df_sim['descuento_porcentaje'] = ((df_sim['precio_venta'] - df_sim['precio_base']) / df_sim['precio_base']) * 100
    
    # Recalcular ratio_precio
    df_sim['ratio_precio'] = df_sim['precio_venta'] / df_sim['precio_competencia']
    
    # Hacer predicciones recursivas
    predicciones, df_sim_actualizado = hacer_predicciones_recursivas(df_sim, modelo)
    
    if predicciones is None:
        return None, None
    
    # Calcular ingresos
    df_sim_actualizado['ingresos_predicho'] = predicciones * df_sim_actualizado['precio_venta']
    
    return predicciones, df_sim_actualizado

# ==================== INTERFAZ PRINCIPAL ====================
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .section-divider {
        margin-top: 30px;
        margin-bottom: 20px;
        border-top: 2px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Cargar modelo y datos
modelo = cargar_modelo()
df = cargar_datos()

if modelo is None:
    st.error("❌ No se pudo cargar el modelo. Verifica que modelo_final.joblib existe en la carpeta 'models'")
    st.stop()

if df is None:
    st.error("❌ No se pudieron cargar los datos. Verifica que inferencia_df_transformado.csv existe en 'data/processed'")
    st.stop()

# ==================== SIDEBAR ====================
st.sidebar.markdown("# 🎮 Controles de Simulación")
st.sidebar.markdown("---")

# Obtener lista única de productos
productos = sorted(df['nombre'].unique())

# Selector de producto
producto_seleccionado = st.sidebar.selectbox(
    "📦 Selecciona un producto:",
    options=productos,
    help="Elige el producto para el cual deseas realizar la simulación"
)

st.sidebar.markdown("---")

# Slider de descuento
descuento_pct = st.sidebar.slider(
    "💰 Ajuste de descuento (%)",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    help="Ajusta el descuento o incremento de precio (-50% a +50%)"
)

st.sidebar.markdown("---")

# Selector de escenario de competencia
escenario_competencia = st.sidebar.radio(
    "🏆 Escenario de competencia:",
    options=["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    help="Selecciona cómo variarán los precios de la competencia"
)

st.sidebar.markdown("---")

# Botón de simulación
if st.sidebar.button("🚀 Simular Ventas", use_container_width=True, type="primary"):
    st.session_state.simular = True

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="padding: 15px; background-color: #f0f2f6; border-radius: 10px;">
        <p style="font-size: 12px; color: #666;">
        💡 <b>Tip:</b> Ajusta los controles y haz clic en "Simular Ventas" para ver 
        las predicciones actualizadas.
        </p>
    </div>
""", unsafe_allow_html=True)

# ==================== ZONA PRINCIPAL ====================
if 'simular' not in st.session_state:
    st.session_state.simular = False

if st.session_state.simular:
    # Header del dashboard
    st.markdown(f"""
        <div class="main-header">
            <h1>📊 Dashboard de Simulación</h1>
            <p style="font-size: 18px; margin: 0;">Noviembre 2025 | <b>{producto_seleccionado}</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Realizar simulación
    predicciones, df_resultados = realizar_simulacion(
        df, 
        producto_seleccionado, 
        descuento_pct, 
        escenario_competencia, 
        modelo
    )
    
    if predicciones is not None:
        # Calcular KPIs
        unidades_totales = np.sum(predicciones)
        ingresos_totales = np.sum(df_resultados['ingresos_predicho'])
        precio_promedio = df_resultados['precio_venta'].mean()
        descuento_promedio = df_resultados['descuento_porcentaje'].mean()
        
        # KPIs destacados
        st.markdown("## 📈 KPIs Destacados")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📦 Unidades Totales",
                value=f"{int(unidades_totales):,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="💵 Ingresos Proyectados",
                value=f"€{ingresos_totales:,.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="💲 Precio Promedio",
                value=f"€{precio_promedio:.2f}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="🏷️ Descuento Promedio",
                value=f"{descuento_promedio:.2f}%",
                delta=None
            )
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Gráfico de predicción diaria
        st.markdown("## 📉 Predicción Diaria de Ventas")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Configurar estilo
        sns.set_style("whitegrid")
        
        # Datos para el gráfico
        dias = df_resultados['dia_mes'].values
        unidades = predicciones
        
        # Línea principal de predicciones
        ax.plot(dias, unidades, linewidth=2.5, color=COLORS['primary'], marker='o', 
                markersize=4, label='Predicción de ventas', zorder=2)
        
        # Marcar Black Friday (día 28)
        idx_bf = np.where(dias == 28)[0]
        if len(idx_bf) > 0:
            idx_bf = idx_bf[0]
            ax.axvline(x=28, color=COLORS['warning'], linestyle='--', linewidth=2.5, alpha=0.7, zorder=1)
            ax.scatter([28], [unidades[idx_bf]], color='red', s=200, zorder=3, edgecolors='darkred', linewidth=2)
            ax.annotate('🛍️ BLACK FRIDAY\n28 NOV', 
                       xy=(28, unidades[idx_bf]), 
                       xytext=(28, unidades[idx_bf] + max(unidades)*0.1),
                       ha='center',
                       fontsize=10,
                       fontweight='bold',
                       color='red',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.set_xlabel('Día de Noviembre', fontsize=12, fontweight='bold')
        ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
        ax.set_title(f'Predicción de Ventas Diarias - {producto_seleccionado}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(1, 31, 2))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=11)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Tabla detallada
        st.markdown("## 📋 Detalle Diario de Predicciones")
        
        # Preparar tabla
        df_tabla = df_resultados[['fecha', 'dia_semana', 'precio_venta', 'precio_competencia', 
                                   'descuento_porcentaje']].copy()
        df_tabla['unidades_predichas'] = predicciones.astype(int)
        df_tabla['ingresos_predichos'] = (predicciones * df_resultados['precio_venta']).round(2)
        
        # Renombrar columnas
        df_tabla.columns = ['Fecha', 'Día Semana', 'Precio Venta (€)', 'Precio Competencia (€)', 
                           'Descuento (%)', 'Unidades', 'Ingresos (€)']
        
        # Formatear
        df_tabla['Fecha'] = df_tabla['Fecha'].dt.strftime('%d-%m-%Y')
        df_tabla['Precio Venta (€)'] = df_tabla['Precio Venta (€)'].apply(lambda x: f"€{x:.2f}")
        df_tabla['Precio Competencia (€)'] = df_tabla['Precio Competencia (€)'].apply(lambda x: f"€{x:.2f}")
        df_tabla['Descuento (%)'] = df_tabla['Descuento (%)'].apply(lambda x: f"{x:.2f}%")
        df_tabla['Ingresos (€)'] = df_tabla['Ingresos (€)'].apply(lambda x: f"€{x:,.2f}")
        
        # Destacar Black Friday
        def highlight_black_friday(row):
            if '28' in row['Fecha']:
                return ['background-color: #FFE5B4'] * len(row)
            return [''] * len(row)
        
        styled_df = df_tabla.style.apply(highlight_black_friday, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Comparativa de escenarios
        st.markdown("## 🎯 Comparativa de Escenarios de Competencia")
        
        escenarios_info = {
            "Actual (0%)": 1.0,
            "Competencia -5%": 0.95,
            "Competencia +5%": 1.05
        }
        
        resultados_escenarios = []
        
        for escenario_nombre, factor in escenarios_info.items():
            # Crear copia para simulación
            df_esc = df[df['nombre'] == producto_seleccionado].copy()
            df_esc = df_esc.sort_values('fecha').reset_index(drop=True)
            
            # Aplicar cambios
            df_esc['precio_venta'] = df_esc['precio_base'] * (1 + descuento_pct / 100)
            
            # Ajustar precios de competencia individuales (con sufijos)
            for col in ['Amazon_x', 'Decathlon_x', 'Deporvillage_x', 'Amazon_y', 'Decathlon_y', 'Deporvillage_y']:
                if col in df_esc.columns:
                    df_esc[col] = df_esc[col] * factor
            
            # Crear columnas Amazon, Decathlon, Deporvillage para el modelo
            for col_base in ['Amazon', 'Decathlon', 'Deporvillage']:
                col_x = f'{col_base}_x'
                col_y = f'{col_base}_y'
                
                if col_x in df_esc.columns and col_y in df_esc.columns:
                    df_esc[col_base] = df_esc[[col_x, col_y]].mean(axis=1)
                elif col_x in df_esc.columns:
                    df_esc[col_base] = df_esc[col_x]
                elif col_y in df_esc.columns:
                    df_esc[col_base] = df_esc[col_y]
            
            df_esc['descuento_porcentaje'] = ((df_esc['precio_venta'] - df_esc['precio_base']) / df_esc['precio_base']) * 100
            df_esc['ratio_precio'] = df_esc['precio_venta'] / df_esc['precio_competencia']
            
            # Predicciones
            preds_esc, df_esc_act = hacer_predicciones_recursivas(df_esc, modelo)
            
            if preds_esc is None:
                continue
            
            # Calcular totales
            unidades_esc = np.sum(preds_esc)
            ingresos_esc = np.sum(preds_esc * df_esc_act['precio_venta'])
            
            resultados_escenarios.append({
                'escenario': escenario_nombre,
                'unidades': unidades_esc,
                'ingresos': ingresos_esc
            })
        
        # Mostrar comparativa en tarjetas
        cols = st.columns(3)
        
        for i, resultado in enumerate(resultados_escenarios):
            with cols[i]:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%); 
                                padding: 20px; border-radius: 10px; color: white; text-align: center;">
                        <h4 style="margin: 0;">{resultado['escenario']}</h4>
                        <p style="font-size: 14px; margin: 10px 0 0 0;">📦 Unidades</p>
                        <p style="font-size: 24px; fontweight: bold; margin: 5px 0;">{int(resultado['unidades']):,}</p>
                        <p style="font-size: 14px; margin: 10px 0 0 0;">💵 Ingresos</p>
                        <p style="font-size: 20px; fontweight: bold; margin: 5px 0;">€{resultado['ingresos']:,.2f}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Mensaje de éxito
        st.success("✅ Simulación completada exitosamente. Los datos están listos para análisis.")
        
else:
    st.markdown("""
        <div class="main-header">
            <h1>📊 Simulador de Ventas Noviembre 2025</h1>
            <p style="font-size: 18px; margin: 0;">Predicciones ML para Noviembre</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="padding: 30px; background-color: #f0f2f6; border-radius: 10px; text-align: center;">
            <h3>👋 ¡Bienvenido al Simulador de Ventas!</h3>
            <p style="font-size: 16px; color: #666; margin-top: 15px;">
            Utiliza los controles del sidebar izquierdo para:
            </p>
            <ul style="text-align: left; display: inline-block; color: #666;">
                <li>📦 Seleccionar un producto</li>
                <li>💰 Ajustar el descuento de precio</li>
                <li>🏆 Elegir un escenario de competencia</li>
                <li>🚀 Ejecutar la simulación</li>
            </ul>
            <p style="font-size: 14px; color: #999; margin-top: 20px;">
            Una vez ejecutada la simulación, verás:</p>
            <ul style="text-align: left; display: inline-block; color: #666; font-size: 14px;">
                <li>📈 KPIs destacados (unidades, ingresos, precios)</li>
                <li>📉 Gráfico de predicción diaria con Black Friday marcado</li>
                <li>📋 Tabla detallada con all los datos diarios</li>
                <li>🎯 Comparativa de escenarios de competencia</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
