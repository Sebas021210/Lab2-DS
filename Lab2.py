# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro

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