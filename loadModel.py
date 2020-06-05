import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import pickle

squareRes = 256
res = [64, 128, 256]
generator = tf.keras.models.load_model("D:\ML\GAN\ganModels\Cancer\chunk1cancerGENModel128x128res-75000epochs-200latent.model")
again = input("Press Enter to see the results: ")
for i in res:
    try:
        while again == "":
            noise = np.random.randn(1, 200)
            plt.imshow(tf.reshape(generator(noise), (i, i)), cmap=plt.cm.bone)
            plt.show()
            again = input("Press Enter to see the results: ")
    except Exception as e:
        print("Wrong Resolution.")
