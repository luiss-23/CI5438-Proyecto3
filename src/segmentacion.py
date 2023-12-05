import numpy as np
import matplotlib.pyplot as plt
from k_means import KMeans
import cv2
import sys

def main():

    # Leer los argumentos de la línea de comandos
    if len(sys.argv) != 3:
        print('Uso: python segmentacion.py <imagen> <k>')
        print('Donde: <imagen> es el nombre de la imagen a segmentar')
        print('       <k> es el número de clusters a generar')
        print('       <k> tambien puede representar un intervalo de valores de k')
        print('Ejemplo: python3 segmentacion.py koi.jpg 3-5')
        print('Ejemplo: python3 segmentacion.py koi.jpg 3')
        sys.exit(1)
    

    nombre_imagen = sys.argv[1]
    k = sys.argv[2]

    if len(k) > 1:
        k_min = int(k[0])
        k_max = int(k[2])+1
        if k_min >= k_max:
            print('Error: El valor de k debe ser un intervalo de valores creciente')
            print('Ejemplo: python3 segmentacion.py koi.jpg 3-5')
            sys.exit(1)
    else:
        k_min = int(k)
        k_max = k_min+1

    # Cargar las imágenes
    imagen = cv2.imread(f'./img/{nombre_imagen}')

    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = imagen.reshape((-1,3))
    
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)

    for k in range(k_min, k_max):
        print(f'\nImagen = {nombre_imagen} ; k = {k}')
        # Ejecutar el algoritmo de k-means
        K_imagen = KMeans(k)
        centroides = K_imagen.k_means(pixel_vals)
        
        # Renderizar las imágenes resultantes
        # Convertir los valores de los centroides a valores enteros de 8 bits
        centroides = np.uint8(centroides)

        labels1 = K_imagen.etiquetar(pixel_vals)
        
        segmented_data1 = centroides[labels1.flatten()]

        segmented_image1 = segmented_data1.reshape((imagen.shape))
        #plt.imshow(segmented_image1)
        plt.imsave(f'./img/k={k}{nombre_imagen}', segmented_image1)

if __name__ == '__main__':
    main()