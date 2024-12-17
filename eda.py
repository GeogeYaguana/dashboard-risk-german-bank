import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
## EDA
df = pd.read_csv('german_credit_data.csv')
df['Risk_encoded'] = df['Risk'].map({'good': 0, 'bad': 1})
## Columnas de la base
print(df.head())
## Mostrar Informacion
def show_info(data):
    print('DATASET SHAPE: ', data.shape, '\n')
    print('-'*50)
    print('FETURE DATA TYPES:')
    print(data.info())
    print('\n', '-'*50)
    print("\nValores únicos de cada columna:",'\n')
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} únicos")
    print('\n', '-'*50)
    print("Valores nulos por columna:")
    print(df.isnull().sum(),'\n')
    print("Duplicados en el dataset:", df.duplicated().sum())
    print("\nResumen estadístico:")
    print(df.describe())
##show_info(df)
def show_corr(data):
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

def show_info_catg(data):
# Distribución de variables categóricas
    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']

    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, data=df)
        plt.title(f'Distribución de {col}')
        plt.xticks(rotation=45)
        plt.show()
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, hue='Risk', data=df)
        plt.title(f'Distribución de {col} por Riesgo')
        plt.xticks(rotation=45)
        plt.show()

##show_info_catg(df)
def show_info_num(data):
    # Histogramas
    numerical_cols = ['Age', 'Credit amount', 'Duration']

    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30, color="skyblue")
        plt.title(f'Histograma de {col}')
        plt.show()

    # Boxplots
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col], color="orange")
        plt.title(f'Boxplot de {col}')
        plt.show()
##show_info_num(df)

def show_risk_relationship(data):
        # Boxplots por riesgo
    numerical_cols = ['Age', 'Credit amount', 'Duration']
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='Risk', y=col, data=df)
        plt.title(f'{col} por nivel de Riesgo')
        plt.show()
## show_risk_relationship(df)

def detectar_aberrantes(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f'Outliers en {col}: {len(outliers)}')

for col in ['Credit amount', 'Duration', 'Age']:
    detectar_aberrantes(col)
sns.pairplot(df, hue="Risk", vars=["Age", "Credit amount", "Duration"])
plt.show()
