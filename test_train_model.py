import tensorflow as tf

mnist = tf.keras.datasets.mnist
x_train, y_train = mnist.load_data()[0]

x_train = x_train / 225
x_train_flattened = x_train.reshape(len(x_train), 28*28)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(784, )))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train_flattened, y_train, epochs=1)
print(model.predict(x_train_flattened))
model.save('handwritten.model')

# image_number = 0
# while os.path.isfile(f"prepared/digit{image_number}.png"):
#     try:
#         img = cv2.imread(f"prepared/digit{image_number}.png")[:,:,0]
#         img = np.invert(np.array([img]))
#         prediction = model.predict(img)
#         print(f"This digit is probably a {np.argmax(prediction)}")
#         plt.imshow(img[0], cmap=plt.cm.binary)
#         plt.show()
#     except:
#         print("Error!")
#     finally:
#         image_number += 1