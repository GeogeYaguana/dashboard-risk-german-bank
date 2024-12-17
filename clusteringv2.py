import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

# Cargar datos
df = pd.read_csv('german_credit_data.csv')

# Codificar la variable objetivo
df['Risk_encoded'] = df['Risk'].map({'good': 0, 'bad': 1})

# Definir columnas numéricas y categóricas
numerical_cols = ['Age', 'Credit amount', 'Duration']
categorical_cols = ['Saving accounts', 'Checking account', 'Housing', 'Purpose']

# Crear el preprocesador con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='drop'  
)

# Definir el Pipeline
clustering_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('clustering', KMeans(n_clusters=4, random_state=42))
])

# Aplicar el Pipeline
df['Cluster'] = clustering_pipeline.fit_predict(df)

# visualizar la distribución de Riesgo por clúster
plt.figure(figsize=(8,4))
sns.countplot(x='Cluster', hue='Risk', data=df, palette='coolwarm')
plt.title('Distribución de Riesgo por Cluster')
plt.show()

# Visualización con PCA 
processed_data = clustering_pipeline.named_steps['preprocessing'].transform(df)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(processed_data)
df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = df['Cluster']

plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette='Set2', s=100)
plt.title('Visualización de Clusters (PCA)')
plt.show()
