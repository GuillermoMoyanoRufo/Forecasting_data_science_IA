# Machine Learning Project

## Project Structure

```
├── data/
│   ├── raw/          # Original, immutable data
│   └── processed/    # Cleaned and processed data
├── notebooks/        # Jupyter notebooks for exploration and analysis
├── models/          # Trained and serialized models
├── app/            # Streamlit application files
├── docs/           # Documentation
└── requirements.txt # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

- Data exploration and model development notebooks are in the `notebooks/` directory
- Raw data should be placed in `data/raw/`
- Processed data will be saved in `data/processed/`
- Trained models are saved in `models/`
- The Streamlit app is in the `app/` directory

## Running the Streamlit App

```bash
cd app
streamlit run app.py
```