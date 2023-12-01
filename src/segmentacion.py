import numpy as np
import matplotlib.pyplot as plt
from k_means import k_means, asignar_cluster
import cv2

def main():
    # Cargar las imágenes
    imagen_1 = cv2.imread('../img/italy.jpg')
    imagen_2 = cv2.imread('../img/koi.jpg')

    imagen_1 = cv2.cvtColor(imagen_1, cv2.COLOR_BGR2RGB)
    imagen_2 = cv2.cvtColor(imagen_2, cv2.COLOR_BGR2RGB)

    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals1 = imagen_1.reshape((-1,3))
    pixel_vals2 = imagen_2.reshape((-1,3))
    
    # Convert to float type
    pixel_vals1 = np.float32(pixel_vals1)
    pixel_vals2 = np.float32(pixel_vals2)

    # Ejecutar el algoritmo de k-means
    k = 3
    centroides_1 = k_means(pixel_vals1, k)
    centroides_2 = k_means(pixel_vals2, k)
    
    # Renderizar las imágenes resultantes
    # convert data into 8-bit values
    centroides_1 = np.uint8(centroides_1)
    centroides_2 = np.uint8(centroides_2)

    labels1 = asignar_cluster(pixel_vals1, centroides_1)
    labels2 = asignar_cluster(pixel_vals2, centroides_2)
    
    segmented_data1 = centroides_1[labels1.flatten()]
    segmented_data2 = centroides_2[labels2.flatten()]

    segmented_image1 = segmented_data1.reshape((imagen_1.shape))
    segmented_image2 = segmented_data2.reshape((imagen_2.shape))
    #print(segmented_image1.shape)
    plt.imshow(segmented_image1)
    plt.show()
    #print(segmented_image2.shape)
    plt.imshow(segmented_image2)
    plt.show()

if __name__ == '__main__':
    main()