# 📈 Demand & Revenue Predictive Simulator - November Sales Season

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ventas-noviembre-simulador.streamlit.app/)

## 📝 Project Overview
This project features an interactive **Business Intelligence and Machine Learning** tool designed to forecast sales demand and revenue for an e-commerce electronics catalog during the critical November peak season.

The core objective is to equip sales teams with a **"What-If" analysis tool**, allowing them to simulate various market scenarios by adjusting pricing strategies and competitor behavior before making real-world execution decisions.

---

## 🛠️ Tech Stack & Key Functionalities

* **Machine Learning:** Random Forest Regressor implementing **Recursive Prediction Logic** (Multi-step forecasting).
* **Feature Engineering:** Advanced time-series processing, including 7-day lags, Moving Averages (MA7), and seasonality factors.
* **Interactive Interface:** Built with **Streamlit**, featuring real-time KPI recalculation and dynamic data visualization using Matplotlib/Seaborn.
* **Data Pipeline:** End-to-end processing, from raw data ingestion and cleaning to feature transformation for model inference.
  
## 📁 Project Structure
```text
├── data/
│   ├── raw/          # Original and immutable data
│   └── processed/    # Clean data and variables generated for inference
├── notebooks/        # EDA (Exploratory Data Analysis) and Model Training
├── models/           # Serialized models ready for production (.joblib)
├── app/              # Streamlit Application (Frontend and Logic)
└── requirements.txt  # Project dependencies
