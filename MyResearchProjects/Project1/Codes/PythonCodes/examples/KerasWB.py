import tensorflow as tf
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt


from keras.layers import Dense
from keras import datasets
from keras.layers import Flatten
from keras.callbacks import TensorBoard
from time import time

model = Sequential()
#x_train = np.array([1.0, 2.0, 3.0, 5.0, 2.0, -1.0], dtype=float)
#y_train = np.array([2.0, 4.0, 6.0, 5.0, 2.0, -2.0], dtype=float)

#model.add(Dense(units=1,input_shape=[1]))
#model.add(layer=Dense(units=1,activation='softmax'))
#model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])

# model.fit(x_train,y_train,batch_size=1,epochs=500)
#
# pred = model.predict(np.array([7.0]))
#
# print(pred)


tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

fashion_mnist = datasets.fashion_mnist
(train_images, train_label), (test_images, test_label) = fashion_mnist.load_data()

train_images = train_images/255
test_images = test_images/255
#plt.imshow(train_images[0])
# plt.imshow(train_images[4])
# plt.show()
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dense(10,activation='softmax'))
model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_label,epochs=5)
model.save_weights('fashion_mnist_1.h5')


# pred_label = model.predict(test_images[0])
# true_label = test_label[0]
# print("Predicted label: ", pred_label)
# print("True Label:", true_label)
model.evaluate(test_images, test_label)


classifications = model.predict(test_images)

print(classifications[0])
print(test_label[0])
# import tensorflow as tf
# import numpy as np
# from tensorflow import keras
# model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# model.compile(optimizer='sgd', loss='mean_squared_error')
# # xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
# # ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
# xs = np.array([1.0, 2.0, 3.0, 5.0, 2.0, -1.0], dtype=float)
# ys = np.array([2.0, 4.0, 6.0, 5.0, 2.0, -2.0], dtype=float)
# model.fit(xs, ys, epochs=1000)
# print(model.predict([7.0]))