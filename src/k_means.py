import numpy as np
from scipy.spatial.distance import cdist
class KMeans:
    
    def __init__(self, k, max_iters=100):
        self.cant_clusters = k
        self.max_iters = max_iters
        self.inercia = 0
        self.centroides = []

    # Función para ejecutar el algoritmo k-means
    def k_means(self,data):

        # Inicializar los centroides de forma aleatoria
        # Se escogen k puntos de forma aleatoria, sin reemplazo a partir de los datos
        self.centroides = data[np.random.choice(data.shape[0], self.cant_clusters, replace=False)]
        prev_centroids = []

        # Iterar hasta que no haya cambios en los centroides
        # O hasta que se alcance el número máximo de iteraciones
        for  iter in range(self.max_iters):
     
            # Asignar cada punto de datos al cluster más cercano
            etiquetas = self.etiquetar(data)

            # Actualizar los centroides
            # Se calcula la media de la cantidad de puntos mas cercanos a cada centroide
            centroides_actualizados = np.array([data[etiquetas == i].mean(axis=0) for i in range(self.cant_clusters)])

            # Verificar convergencia
            # Comparando con los centroides anteriores
            if iter > 0:
                if np.all(prev_centroids == centroides_actualizados):
                    print(f"Convergencia alcanzada en la iteración {iter}")
                    break
            
            prev_centroids = self.centroides
            self.centroides = centroides_actualizados

        return self.centroides

    # Función para asignar cada punto de datos al cluster más cercano
    def etiquetar(self, data):
            
        # Se calcula la norma euclidiana entre cada punto y cada centroide
        # Nos apoyamos del metodo cdist de scipy
        distancias = cdist(data, self.centroides ,'euclidean')

        # Se calcula la inercia, aprovechando que ya se tienen las distancias
        # La inercia luego se utiliza para aplicar el metodo del codo
        # y poder determinar la cantidad de clusters k optima para el problema
        self.inercia = np.sum(np.min(distancias,axis=1)**2)

        # Se asigna el cluster con la distancia más corta a cada punto
        return np.argmin(distancias, axis=1)
