import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importar os dados do CSV
df = pd.read_csv('medical_examination.csv')

# Adicionar a coluna 'overweight'
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

# Normalizar os dados de colesterol e glicose (0 sempre bom, 1 sempre ruim)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# Função para desenhar o gráfico categórico
def draw_cat_plot():
    # Convertendo os dados para formato longo
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Agrupando e reformando os dados
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Desenhando o gráfico categórico
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig

    return fig

# Função para desenhar o mapa de calor
def draw_heat_map():
    # Filtrar os dados incorretos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculando a matriz de correlação
    corr = df_heat.corr()

    # Gerando uma máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Configurando a figura matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))

    # Desenhando o mapa de calor
    sns.heatmap(corr, annot=True, mask=mask, square=True, fmt='.1f', center=0, vmin=-0.1, vmax=0.25, cmap='coolwarm', ax=ax)

    return fig
