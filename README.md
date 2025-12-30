# 📈 Predictive Demand & Revenue Simulator - November Campaign

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ventas-noviembre-simulador.streamlit.app/)

## 📝 Project Overview
This project features an interactive **Business Intelligence & Machine Learning tool** designed to forecast sales demand and revenue for an electronics e-commerce catalog during the high-stakes November campaign.

The core objective is to empower sales teams with a **"What-If" Analysis tool**, allowing them to simulate market scenarios by adjusting pricing strategies and competitor behavior before making real-world decisions.

## 🔗 Live Demo
**Access the interactive dashboard here:** [https://forecastingdatascienceia-ibqwbjagxbykabcg2gryjd.streamlit.app/](https://forecastingdatascienceia-ibqwbjagxbykabcg2gryjd.streamlit.app/)

---

## 🛠️ Tech Stack & Key Features
* **Machine Learning:** Random Forest Regressor with **Recursive Prediction Logic** (Multi-step forecasting).
* **Feature Engineering:** Time-series processing including 7-day Lags, Moving Averages (MA7), and Seasonality factors.
* **Interactive UI:** Built with **Streamlit**, featuring real-time KPI recalculation and dynamic data visualization with Matplotlib/Seaborn.
* **Data Pipeline:** Full processing from raw cleaning to feature transformation.

## 📁 Project Structure
```text
├── data/
│   ├── raw/          # Original, immutable data
│   └── processed/    # Cleaned data and engineered features for inference
├── notebooks/        # EDA (Exploratory Data Analysis) and Model Training
├── models/           # Serialized production-ready models (.joblib)
├── app/              # Streamlit application (Frontend & Logic)
└── requirements.txt  # Project dependencies

## 🚀 Getting Started

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# From the root directory
streamlit run app/app.py

📊 Business Logic Implemented
Recursive Forecasting: Unlike static models, this app updates its own future "lags" based on previous predictions to maintain accuracy over a 30-day horizon.

Competitor Scenarios: Users can toggle between three competitive intensity levels, impacting the price-ratio and estimated demand.

Black Friday Impact: Special weighting and visual markers for high-traffic dates (Nov 28th).

Author: Guillermo Moyano Rufo

Data Science & AI Project
