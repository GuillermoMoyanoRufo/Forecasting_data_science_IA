import joblib
import pandas as pd

# Cargar modelo
modelo = joblib.load('models/modelo_final.joblib')
print('COLUMNAS ESPERADAS POR EL MODELO:')
print(list(modelo.feature_names_in_))
print('\n' + '='*80)

# Cargar datos de inferencia
df = pd.read_csv('data/processed/inferencia_df_transformado.csv')
print('\nCOLUMNAS DEL DATAFRAME DE INFERENCIA:')
print(list(df.columns))
print('\n' + '='*80)

# Encontrar las diferencias
cols_modelo = set(modelo.feature_names_in_)
cols_df = set(df.columns)

print('\nCOLUMNAS EN MODELO PERO NO EN DF:')
print(sorted(cols_modelo - cols_df))

print('\nCOLUMNAS EN DF PERO NO EN MODELO:')
print(sorted(cols_df - cols_modelo))
