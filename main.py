import cv2
import numpy as np
import matplotlib.pyplot as plt

image_target = cv2.cvtColor(cv2.imread("redim.jpg"), cv2.COLOR_BGR2RGB)
image_source= cv2.cvtColor(cv2.imread("8733654151_b9422bb2ec_k.jpg"), cv2.COLOR_BGR2RGB)
vector_source = image_source.reshape((-1, 3))
vector_target = image_target.reshape((-1, 3))

for i in range(30):

    gaussian = np.random.randn(vector_source.shape[1])#vecteur aléatoire suivant la loi normale(0,1)
    gaussian = gaussian/np.linalg.norm(gaussian)#normalisation du vecteur

    #produit vectoriel itératif entre le vecteur aléatoire et les vecteurs des images
    ready_to_sort_source= np.dot(vector_source, gaussian)
    ready_to_sort_target= np.dot(vector_target, gaussian)

    #formation de doublets (valeur, indice)
    indices = np.arange(ready_to_sort_source.shape[0])
    tab_source = np.vstack((ready_to_sort_source, indices))
    tab_target = np.vstack((ready_to_sort_target, indices))

    #tri des doublets selon la valeur
    reconstructed_source = np.argsort(tab_source[0, :])
    reconstructed_target = np.argsort(tab_target[0, :])
    sorted_source = tab_source[:,reconstructed_source]
    sorted_target = tab_target[:,reconstructed_target]


    indice = sorted_source[1, :].astype(int)

    diff = (sorted_target[0, :] - sorted_source[0, :]) * gaussian[:, np.newaxis]
    vector_source = vector_source.astype('float64')
    vector_source[indice, :] += diff.T
    vector_source = np.clip(vector_source, 0, 255)
    vector_source = vector_source.astype('uint8')
    image_source = vector_source.reshape(image_source.shape)
    plt.imshow(image_source)
    plt.show()

cv2.imwrite("result2.jpg", cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR))






