import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('handwritten.model')


image_number = 0
while os.path.isfile(f"prepared/digit{image_number}.png"):
    try:
        img = cv2.imread(f"prepared/digit{image_number}.png")[:,:,0]
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1