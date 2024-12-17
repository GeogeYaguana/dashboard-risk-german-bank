import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Datos de ejemplo
data = {
    'precio': [200000, 250000, 300000, 400000, 500000],
    'tamaño': [100, 150, 200, 250, 300],
    'habitaciones': [2, 3, 4, 4, 5],
    'baños': [1, 2, 2, 3, 3],
    'edad_casa': [10, 15, 20, 25, 30],
    'ingresos_zona': [50, 60, 70, 80, 90]  # Ingresos promedio de la zona
}

# Crear DataFrame
df = pd.DataFrame(data)

# Calcular la matriz de correlación
correlacion = df.corr()

# Mostrar la matriz de correlación
print("Matriz de correlación:")
print(correlacion)

# Visualizar la matriz de correlación con un heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación para Selección de Características")
plt.show()
