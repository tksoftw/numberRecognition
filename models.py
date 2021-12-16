import os
import cv2
import numpy as np
import tensorflow as tf

def loadModel(model_name: str):
    return tf.keras.models.load_model(model_name)

def getLossAndAccuracy(model) -> tuple:
    mnist = tf.keras.datasets.mnist
    x_test, y_test = mnist.load_data()[1]
    
    x_test = x_test / 225
    x_test_flattened = x_test.reshape(len(x_test), 28*28)
    return model.evaluate(x_test_flattened, y_test)
    
def trainNewModel(epoch_count: int, save='') -> None:
    mnist = tf.keras.datasets.mnist
    x_train, y_train = mnist.load_data()[0]

    x_train = x_train / 225
    x_train_flattened = x_train.reshape(len(x_train), 28*28)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(784,)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x_train_flattened, y_train, epochs=epoch_count)
    if save:
        model.save(save)
    return model

def getPrediction(model, img_path: str) -> int:
    if not os.path.isfile(img_path):
        raise FileNotFoundError
    try:
        img = cv2.imread(img_path)[:,:,0]
        img = np.invert(np.array([img])) # Invert color-scheme
        img = img / 255
        img_flattened = img.reshape(len(img), 28*28)
        prediction = model.predict(img_flattened)
        return prediction[0]
    except Exception:
        raise