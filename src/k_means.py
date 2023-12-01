import numpy as np

def k_means(data, k, max_iters=100):

    # Inicializar los centroides de forma aleatoria
    centroids = data[np.random.randint(0, len(data), k)]
    #centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    prev_centroids = []

    # Iterar hasta que no haya cambios en los centroides
    # O hasta que se alcance el número máximo de iteraciones
    for _ in range(max_iters):

        # Asignar cada punto de datos al cluster más cercano
        # Cada cluster se representa como una lista
        clusters = [[] for _ in range(k)]
        for point in data:
            closest_centroid_index = np.argmin(np.linalg.norm(centroids - point, axis=1))
            clusters[closest_centroid_index].append(point)

        # Actualizar los centroides
        # Se calcula la media de los valores en cada cluster
        for i, cluster in enumerate(clusters):
            centroids[i] = np.mean(cluster, axis=0)
        
        # Verificar convergencia
        # comparando con los centroides anteriores
        if np.all(prev_centroids == centroids):
            break
        
        prev_centroids = centroids

    return centroids

# Función para asignar cada punto de datos al cluster más cercano
def asignar_cluster(data, centroides):
        
        # Recorre cada punto de datos y calcula la distancia con cada centroide
        distancias = np.linalg.norm(data[:, np.newaxis] - centroides, axis=2)
        
        # Se asigna el cluster con la distancia más corta a cada punto
        return np.argmin(distancias, axis=1)
