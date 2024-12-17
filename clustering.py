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

# Variables numéricas
numerical_cols = ['Age', 'Credit amount', 'Duration']

# Escalamiento de variables numéricas
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numerical_cols])

# PCA con las variables numéricas (opcional, para visualización inicial)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Visualización del PCA coloreado por el riesgo
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['Risk'], palette="coolwarm")
plt.title('PCA de variables numéricas')
plt.show()

# Preprocesamiento completo (numéricas + categóricas)
categorical_cols = ['Saving accounts', 'Checking account', 'Housing', 'Purpose']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

# Aplicar transformación
processed_data = preprocessor.fit_transform(df)

# Determinación de k usando el método del codo
inertia = []
k_range = range(2, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(processed_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Método del Codo para determinar k')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.show()

# Suponiendo que el mejor k es 4 (ajustar según la gráfica del codo)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(processed_data)

# PCA para visualización de los clusters
pca = PCA(n_components=2)
pca_result = pca.fit_transform(processed_data)

df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = df['Cluster']

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette='Set2', s=100)
plt.title('Visualización de Clusters (PCA)')
plt.show()

# Resumen de los clusters
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
print(cluster_summary)

# Gráfico de la distribución de Riesgo por Cluster
plt.figure(figsize=(8, 4))
sns.countplot(x='Cluster', hue='Risk', data=df, palette='coolwarm')
plt.title('Distribución de Riesgo por Cluster')
plt.show()

clustering_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('clustering', KMeans(n_clusters=4, random_state=42))
])

# Aplicar pipeline
df['Cluster'] = clustering_pipeline.fit_predict(df)
