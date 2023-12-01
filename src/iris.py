import pandas as pd
from k_means import k_means, asignar_cluster
import matplotlib.pyplot as plt

def main():
    
    
    # Leer archivo
    # No se toma en cuenta la columna de especies
    # Para trabajar con data no etiquetada
    data_iris = pd.read_csv("../docs/iris.csv", sep=',')
    X = data_iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]    
    y = data_iris["species"]

    """centroides = []
    clusters = []
    for k in range(1,21):
        clus , cent = k_means(X.values, k, 100)
        centroides.append(cent)
        clusters.append(clus)"""
    
    k = 10
    
    clusters, centroides = k_means(X.values, k, 100)

    plt.scatter(X.values[:,0], X.values[:,1], s = 50, label='Datos')
    plt.scatter(centroides[:,0], centroides[:,1], c='red', marker='*', s=100, label='Centroides')
    plt.legend()
    plt.title(f'K-Means con k: {k}')
    plt.show()

    etiquetas = asignar_cluster(X.values,centroides)    

    species_to_int = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    y_test = [species_to_int[x] for x in y]

    for i in range(len(etiquetas)):
        print(f"Kmeans: {etiquetas[i]}, real: {y_test[i]}")
    

if __name__ == '__main__':
    main()