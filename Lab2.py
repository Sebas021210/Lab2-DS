# %%
"""
# Laboratorio No. 2 - Data Science
"""

# %%
"""
Manuel Rodas - 21509 / Sebastián Solorzano - 21826
"""

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from prophet import Prophet
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# %%
"""
### Carga de datos
"""

# %%
def limpiar_datos(df):
    for col in df.columns[1:]:
        df[col] = df[col].str.replace(' ', '', regex=True)
        df[col] = df[col].str.replace(',', '.', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# %%
consumos_df = pd.read_csv('consumo.csv', sep=';', encoding='utf-8')
consumos_df = consumos_df.dropna(axis=1, how='all')
consumos_df = consumos_df[['Fecha', 'Gasolina regular', 'Gasolina superior', 'Diesel bajo azufre', 'Gas licuado de petr�leo']]
consumos_df = limpiar_datos(consumos_df)
consumos_df.head()

# %%
importaciones_df1 = pd.read_csv('importacion.csv', sep=';', encoding='utf-8')
importaciones_df1 = importaciones_df1.dropna(axis=1, how='all')
importaciones_df1 = importaciones_df1[['Fecha', 'Gasolina regular', 'Gasolina superior', 'Diesel bajo azufre', 'Gas licuado de petr�leo']]
importaciones_df1 = limpiar_datos(importaciones_df1)
importaciones_df1.head()

# %%
precios2024_df = pd.read_csv('precios2024.csv', sep=';', encoding='utf-8')
precios2024_df = precios2024_df.dropna(axis=1, how='all')

precios2023_df = pd.read_csv('precios2023.csv', sep=';', encoding='utf-8')
precios2023_df = precios2023_df.dropna(axis=1, how='all')

precios2022_df = pd.read_csv('precios2022.csv', sep=';', encoding='utf-8')
precios2022_df = precios2022_df.dropna(axis=1, how='all')

precios2021_df = pd.read_csv('precios2021.csv', sep=';', encoding='utf-8')
precios2021_df = precios2021_df.dropna(axis=1, how='all')

precios_df = pd.concat([precios2021_df, precios2022_df, precios2023_df, precios2024_df])
precios_df = precios_df[['FECHA', 'Superior', 'Regular', 'Diesel', 'Glp Cilindro 25Lbs.']]
precios_df.columns = ['Fecha', 'Gasolina superior', 'Gasolina regular', 'Diesel', 'Gas licuado de petróleo']
precios_df = limpiar_datos(precios_df)
precios_df.head()


# %%
"""
### Análisis exploratorio
"""

# %%
def verificar_normalidad(df):
    resultados = {}
    for col in df.columns[1:]:
        stat, p_value = shapiro(df[col].dropna())
        resultados[col] = p_value
    return resultados

# %%
def graficar_datos(consumos_df, importaciones_df1, precios_df):
    plt.figure(figsize=(14, 10))
    
    # Graficar datos de consumo
    plt.subplot(3, 1, 1)
    consumos_df[['Fecha', 'Gasolina regular', 'Gasolina superior', 'Diesel bajo azufre', 'Gas licuado de petr�leo']].set_index('Fecha').plot(kind='line', ax=plt.gca())
    plt.title('Consumo de Combustibles')
    plt.xticks(rotation=45)
    
    # Graficar datos de importación
    plt.subplot(3, 1, 2)
    importaciones_df1[['Fecha', 'Gasolina regular', 'Gasolina superior', 'Diesel bajo azufre', 'Gas licuado de petr�leo']].set_index('Fecha').plot(kind='line', ax=plt.gca())
    plt.title('Importaciones de Combustibles')
    plt.xticks(rotation=45)
    
    # Graficar datos de precios
    plt.subplot(3, 1, 3)
    precios_df[['Fecha', 'Gasolina superior', 'Gasolina regular', 'Diesel', 'Gas licuado de petróleo']].set_index('Fecha').plot(kind='line', ax=plt.gca())
    plt.title('Precios de Combustibles')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analizar_picos(importaciones_df1):    
    importaciones_df2 = importaciones_df1.copy()

    # Convertir la columna 'Fecha' a datetime y establecerla como índice
    importaciones_df2['Fecha'] = pd.to_datetime(importaciones_df2['Fecha'], format='%b/%Y', errors='coerce')
    
    # Verificar si la conversión de fechas fue exitosa
    if importaciones_df2['Fecha'].isnull().all():
        print("Error: No se pudieron convertir las fechas. Verifique el formato de las fechas en el CSV.")
        return
    
    importaciones_df2.set_index('Fecha', inplace=True)
    
    # Agrupar datos por mes manualmente
    importaciones_mensuales = importaciones_df2[['Gasolina regular', 'Gasolina superior', 'Diesel bajo azufre', 'Gas licuado de petr�leo']].resample('M').sum()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=importaciones_mensuales)
    plt.title('Importaciones Mensuales de Combustibles')
    plt.xlabel('Fecha')
    plt.ylabel('Volumen')
    plt.xticks(rotation=45, ha='right')
    plt.show()
        
    # Filtrar por período de pandemia
    importaciones_pandemia = importaciones_mensuales.loc['2020-03':'2022-12']
    
    # Verificar si hay datos durante la pandemia
    if importaciones_pandemia.empty:
        print("No hay datos disponibles para el período de la pandemia.")
    else:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=importaciones_pandemia)
        plt.title('Importaciones durante la Pandemia')
        plt.xlabel('Fecha')
        plt.ylabel('Volumen')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()  # Ajustar el diseño para que no se corten etiquetas
        plt.show()

# %%
normalidad_consumo = verificar_normalidad(consumos_df)
normalidad_importacion = verificar_normalidad(importaciones_df1)
normalidad_precios = verificar_normalidad(precios_df)

print("Normalidad en datos de consumo:", normalidad_consumo)
print("Normalidad en datos de importación:", normalidad_importacion)
print("Normalidad en datos de precios:", normalidad_precios)

# %%
graficar_datos(consumos_df, importaciones_df1, precios_df)

# %%
analizar_picos(importaciones_df1)

# %%
"""
### Serie de consumos
"""

# %%
consumos_df2 = consumos_df.copy()
# Convertir la columna 'Fecha' a datetime y establecerla como índice
consumos_df2['Fecha'] = pd.to_datetime(consumos_df2['Fecha'], format='%b/%Y', errors='coerce')

# Verificar si la conversión de fechas fue exitosa
if consumos_df2['Fecha'].isnull().all():
    print("Error: No se pudieron convertir las fechas. Verifique el formato de las fechas en el CSV.")

consumos_df2.set_index('Fecha', inplace=True)

# %%
# Inicio, fin y frecuencia de los datos
print("Inicio de los datos:", consumos_df2.index.min())
print("Fin de los datos:", consumos_df2.index.max())
print("Frecuencia de los datos:", consumos_df2.index.to_series().diff().value_counts())

# %%
# Gráfico de la Serie consumos
plt.figure(figsize=(12, 6))
sns.lineplot(data=consumos_df2.resample('M').sum())
plt.title('Consumo Mensual de Combustibles')
plt.xlabel('Fecha')
plt.ylabel('Volumen')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
# Descomposición de la Serie
consumo_mensual = consumos_df2.resample('M').sum()
consumo_mensual_decomp = consumo_mensual.rolling(window=12).mean()

plt.figure(figsize=(12, 6))
plt.plot(consumo_mensual, label='Serie Original')
plt.plot(consumo_mensual_decomp, label='Tendencia (12-mos)', color='red')
plt.legend()
plt.title('Descomposición de Consumo Mensual')
plt.xlabel('Fecha')
plt.ylabel('Volumen')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
# Transformación de la Serie
consumo_mensual_log = np.log(consumo_mensual.replace(0, np.nan))

# %%
# Estacionariedad en Media
consumo_mensual_columna = consumo_mensual['Gasolina regular']

# Gráfico de ACF
plt.figure(figsize=(12, 6))
plot_acf(consumo_mensual_columna.dropna(), lags=50, ax=plt.gca())
plt.title('ACF de Consumo Mensual')
plt.show()

# Prueba de Estacionariedad ADF
adf_result = adfuller(consumo_mensual_columna.dropna())
print(f'Estadístico ADF: {adf_result[0]}')
print(f'Valor p: {adf_result[1]}')

# %%
# Estacionariedad en Media
consumo_mensual_columna = consumo_mensual['Gasolina regular'].dropna()

# Parámetros del Modelo ARIMA
auto_model = auto_arima(consumo_mensual_columna, seasonal=True, m=12)
print(auto_model.summary())

# %%
# Estacionariedad en Media
consumo_mensual_columna = consumo_mensual['Gasolina regular'].dropna()

# Modelos ARIMA
best_model = ARIMA(consumo_mensual_columna, order=auto_model.order)
best_fit = best_model.fit()

plt.figure(figsize=(12, 6))
plt.plot(best_fit.fittedvalues, label='Fitted Values')
plt.plot(consumo_mensual_columna, label='Actual Values')
plt.legend()
plt.title('Modelo ARIMA - Consumo Mensual')
plt.xlabel('Fecha')
plt.ylabel('Volumen')
plt.xticks(rotation=45, ha='right')
plt.show()

print(f'AIC: {best_fit.aic}')
print(f'BIC: {best_fit.bic}')

# %%
# Crear el DataFrame para Prophet
consumo_df_prophet = consumos_df[['Fecha', 'Gasolina regular']].copy()
consumo_df_prophet['Fecha'] = pd.to_datetime(consumo_df_prophet['Fecha'], format='%b/%Y', errors='coerce')
consumo_df_prophet = consumo_df_prophet.dropna(subset=['Fecha'])  # Eliminar filas con fechas inválidas
consumo_df_prophet = consumo_df_prophet.rename(columns={'Fecha': 'ds', 'Gasolina regular': 'y'})
consumo_df_prophet = consumo_df_prophet.dropna(subset=['y'])  # Eliminar filas con valores 'y' NaN

# Modelo Prophet
prophet_model = Prophet()
prophet_model.fit(consumo_df_prophet)

# DataFrame para el futuro
future = prophet_model.make_future_dataframe(periods=12, freq='M')
forecast = prophet_model.predict(future)

# Gráfico del pronóstico
fig = prophet_model.plot(forecast)
plt.title('Modelo Prophet - Consumo Mensual')
plt.show()

# %%
"""
### Serie de importaciones
"""

# %%
importaciones_df2 = importaciones_df1.copy()

importaciones_df2['Fecha'] = pd.to_datetime(importaciones_df2['Fecha'], format='%b/%Y', errors='coerce')
importaciones_df2.set_index('Fecha', inplace=True)

# %%
# Inicio, fin y frecuencia de los datos
print("Inicio de los datos:", importaciones_df2.index.min())
print("Fin de los datos:", importaciones_df2.index.max())
print("Frecuencia de los datos:", importaciones_df2.index.to_series().diff().value_counts())

# %%
# Gráfico de la Serie
plt.figure(figsize=(12, 6))
sns.lineplot(data=importaciones_df2.resample('M').sum())
plt.title('Importaciones Mensuales de Combustibles')
plt.xlabel('Fecha')
plt.ylabel('Volumen')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
# Descomposición de la Serie
importaciones_mensuales = importaciones_df2.resample('M').sum()
importaciones_mensuales_decomp = importaciones_mensuales.rolling(window=12).mean()

plt.figure(figsize=(12, 6))
plt.plot(importaciones_mensuales, label='Serie Original')
plt.plot(importaciones_mensuales_decomp, label='Tendencia (12-mos)', color='red')
plt.legend()
plt.title('Descomposición de Importaciones Mensuales')
plt.xlabel('Fecha')
plt.ylabel('Volumen')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
# Transformación de la Serie
importaciones_mensuales_log = np.log(importaciones_mensuales.replace(0, np.nan))

# %%
importaciones_mensuales_columna = importaciones_mensuales['Gasolina regular']

# Estacionariedad en Media
plt.figure(figsize=(12, 6))
plot_acf(importaciones_mensuales_columna.dropna(), lags=50, ax=plt.gca())
plt.title('ACF de Importaciones Mensuales')
plt.show()

adf_result = adfuller(importaciones_mensuales_columna.dropna())
print(f'Estadístico ADF: {adf_result[0]}')
print(f'Valor p: {adf_result[1]}')

# %%
# Parametros del Modelo ARIMA
auto_model = auto_arima(importaciones_mensuales_columna, seasonal=True, m=12)
print(auto_model.summary())

# %%
# Modelos ARIMA
best_model = ARIMA(importaciones_mensuales_columna, order=auto_model.order)
best_fit = best_model.fit()

plt.figure(figsize=(12, 6))
plt.plot(best_fit.fittedvalues, label='Fitted Values')
plt.plot(importaciones_mensuales_columna, label='Actual Values')
plt.legend()
plt.title('Modelo ARIMA - Importaciones Mensuales')
plt.xlabel('Fecha')
plt.ylabel('Volumen')
plt.xticks(rotation=45, ha='right')
plt.show()

print(f'AIC: {best_fit.aic}')
print(f'BIC: {best_fit.bic}')

# %%
# Modelo Prophet
importaciones_df_prophet = importaciones_df1[['Fecha', 'Gasolina regular']].copy()
importaciones_df_prophet['Fecha'] = pd.to_datetime(importaciones_df_prophet['Fecha'], format='%b/%Y', errors='coerce')
importaciones_df_prophet = importaciones_df_prophet.dropna(subset=['Fecha'])  # Eliminar filas con fechas inválidas
importaciones_df_prophet = importaciones_df_prophet.rename(columns={'Fecha': 'ds', 'Gasolina regular': 'y'})
importaciones_df_prophet = importaciones_df_prophet.dropna(subset=['y'])  # Eliminar filas con valores 'y' NaN

# Modelo Prophet
prophet_model = Prophet()
prophet_model.fit(importaciones_df_prophet)
future = prophet_model.make_future_dataframe(periods=12, freq='M')
forecast = prophet_model.predict(future)

# Gráfico del pronóstico
fig = prophet_model.plot(forecast)
plt.title('Modelo Prophet - Importaciones Mensuales')
plt.show()

# %%
"""
### Serie de precios
"""

# %%
precios_df2 = precios_df.copy()

# Definir un mapa de meses en español a inglés
month_translation = {
    'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
    'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
    'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
}

precios_df2['Fecha'] = precios_df2['Fecha'].str.lower().replace(month_translation, regex=True)
precios_df2['Fecha'] = pd.to_datetime(precios_df2['Fecha'], format='%d/%b/%Y', errors='coerce')
precios_df2.set_index('Fecha', inplace=True)

# %%
# Inicio, fin y frecuencia de los datos
print("Inicio de los datos:", precios_df2.index.min())
print("Fin de los datos:", precios_df2.index.max())
print("Frecuencia de los datos:", precios_df2.index.to_series().diff().value_counts())

# %%
# Gráfico de la Serie de Precios
plt.figure(figsize=(12, 6))
sns.lineplot(data=precios_df2.resample('D').mean())
plt.title('Precios Diarios de Combustibles')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
# Descomposición de la Serie
precios_diarios = precios_df2.resample('D').mean()
precios_diarios_decomp = precios_diarios.rolling(window=30).mean()

plt.figure(figsize=(12, 6))
plt.plot(precios_diarios, label='Serie Original')
plt.plot(precios_diarios_decomp, label='Tendencia (30-días)', color='red')
plt.legend()
plt.title('Descomposición de Precios Diarios')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
# Transformación de la Serie
precios_diarios_log = np.log(precios_diarios.replace(0, np.nan))

# %%
# Estacionariedad en Media
precios_diarios_columna = precios_diarios['Gasolina regular']
plt.figure(figsize=(12, 6))
plot_acf(precios_diarios_columna.dropna(), lags=50, ax=plt.gca())
plt.title('ACF de Precios Diarios')
plt.show()

# Prueba de Estacionariedad ADF
adf_result = adfuller(precios_diarios_columna.dropna())
print(f'Estadístico ADF: {adf_result[0]}')
print(f'Valor p: {adf_result[1]}')

# %%
# Parámetros del Modelo ARIMA
auto_model = auto_arima(precios_diarios_columna, seasonal=True, m=12)
print(auto_model.summary())

# %%
# Modelos ARIMA
best_model = ARIMA(precios_diarios_columna, order=auto_model.order)
best_fit = best_model.fit()

plt.figure(figsize=(12, 6))
plt.plot(best_fit.fittedvalues, label='Fitted Values')
plt.plot(precios_diarios_columna, label='Actual Values')
plt.legend()
plt.title('Modelo ARIMA - Precios Diarios')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.xticks(rotation=45, ha='right')
plt.show()

print(f'AIC: {best_fit.aic}')
print(f'BIC: {best_fit.bic}')

# %%
# Modelo Prophet
precios_df_prophet = precios_df[['Fecha', 'Gasolina regular']].copy()
precios_df_prophet['Fecha'] = precios_df_prophet['Fecha'].str.lower().replace(month_translation, regex=True)
precios_df_prophet['Fecha'] = pd.to_datetime(precios_df_prophet['Fecha'], format='%d/%b/%Y', errors='coerce')
precios_df_prophet = precios_df_prophet.dropna(subset=['Fecha'])  # Eliminar filas con fechas inválidas
precios_df_prophet = precios_df_prophet.rename(columns={'Fecha': 'ds', 'Gasolina regular': 'y'})
precios_df_prophet = precios_df_prophet.dropna(subset=['y'])  # Eliminar filas con valores 'y' NaN

# Modelo Prophet
prophet_model = Prophet()
prophet_model.fit(precios_df_prophet)
future = prophet_model.make_future_dataframe(periods=30, freq='D')
forecast = prophet_model.predict(future)

# Gráfico del pronóstico
fig = prophet_model.plot(forecast)
plt.title('Modelo Prophet - Precios Diarios')
plt.show()

# %%
"""
### Predicciones serie de consumos
"""
