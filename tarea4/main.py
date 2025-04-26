import matplotlib.pyplot as plt
import numpy as np
# Usa la ruta absoluta (ejemplo para Windows)
ruta_archivo = r'C:\Users\matia\projects\ia\datos_evolucion.txt'

# Carga los datos
generaciones, mejores, promedios = np.loadtxt(ruta_archivo, unpack=True)

# Configurar gráfico
plt.figure(figsize=(10, 6))
plt.plot(generaciones, mejores, label='Mejor distancia', marker='o', linestyle='-')
plt.plot(generaciones, promedios, label='Distancia promedio', marker='x', linestyle='--')

# Personalizar
plt.title('Evolución del Algoritmo Genético para TSP')
plt.xlabel('Generación')
plt.ylabel('Distancia')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Guardar y mostrar
plt.savefig('evolucion_tsp.png')
plt.show()
