# 📈 Simulador Predictivo de Demanda e Ingresos - Campaña de Noviembre

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ventas-noviembre-simulador.streamlit.app/)

## 📝 Descripción General del Proyecto
Este proyecto presenta una herramienta interactiva de Business Intelligence y Machine Learning diseñada para pronosticar la demanda de ventas y los ingresos de un catálogo de comercio electrónico de electrónica durante la crítica campaña de noviembre.

El objetivo central es empoderar a los equipos de ventas con una herramienta de análisis "What-If" (¿Qué pasaría si...?), permitiéndoles simular diversos escenarios de mercado mediante el ajuste de estrategias de precios y el comportamiento de la competencia antes de tomar decisiones en el mundo real.

---

## 🛠️ Stack Tecnológico y Funcionalidades Clave

* **Machine Learning:** Random Forest Regressor con **Lógica de Predicción Recursiva** (Pronóstico de múltiples pasos).
* **Ingeniería de Variables:** Procesamiento de series temporales que incluye retardos de 7 días (Lags), Medias Móviles (MA7) y factores de estacionalidad.
* **Interfaz Interactiva:** Desarrollada con **Streamlit**, con recálculo de KPIs en tiempo real y visualización dinámica de datos con Matplotlib/Seaborn.
* **Pipeline de Datos:** Procesamiento completo, desde la limpieza de datos brutos hasta la transformación de variables para inferencia.
  
## 📁 Project Structure
```text
├── data/
│   ├── raw/          # Datos originales e inmutables
│   └── processed/    # Datos limpios y variables generadas para inferencia
├── notebooks/        # EDA (Análisis Exploratorio de Datos) y Entrenamiento del Modelo
├── models/           # Modelos serializados listos para producción (.joblib)
├── app/              # Aplicación Streamlit (Frontend y Lógica)
└── requirements.txt  # Dependencias del proyecto
