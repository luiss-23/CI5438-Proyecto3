import pandas as pd
from k_means import KMeans
import matplotlib.pyplot as plt

def main():
    
    # Leer archivo
    # No se toma en cuenta la columna de especies
    # Para trabajar con data no etiquetada
    data_iris = pd.read_csv("../docs/iris.csv", sep=',')
    X = data_iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]    
    y = data_iris["species"]
    
    # Definir los valores de k a evaluar
    k_values = [2, 3, 4, 5, 6, 7]

    # Ejecutar el algoritmo k-means para cada valor de k
    inertias = []
    for k in k_values:
        print(f"\nk = {k}")
        K_iris = KMeans(k, 100)
        K_iris.k_means(X.values)
        inertias.append(K_iris.inercia)

    # Graficar la inercia en función de k
    # Aplicando el "metodo del codo" se puede apreciar que el k optimo se encuentra entre 3 y 4
    # Esto concuerda con la cantidad de especies de iris que hay en el dataset, que son 3
    plt.plot(k_values, inertias)
    plt.title("Metodo del codo")
    plt.xlabel("k")
    plt.ylabel("Inercia")
    plt.show()
    
    
    k = 3
    K_iris = KMeans(k, 100)
    centroides = K_iris.k_means(X.values)

    plt.scatter(X.values[:,0], X.values[:,1], s = 50, label='Datos')
    plt.scatter(centroides[:,0], centroides[:,1], c='red', marker='*', s=100, label='Centroides')
    plt.legend()
    plt.title(f'Algoritmo K-Means con k: {k}')
    plt.show()

    etiquetas = K_iris.etiquetar(X.values)    

    species_to_int = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    y_test = [species_to_int[x] for x in y]

    # Calcular precision y mostrar resultaods
    correctos = 0
    for i in range(len(etiquetas)):
        print(f"Kmeans: {etiquetas[i]}, real: {y_test[i]}")
        if etiquetas[i] == y_test[i]:
            correctos += 1
    precision = correctos / len(etiquetas)
    print(f"Precision: {precision*100}%")

    

if __name__ == '__main__':
    main()