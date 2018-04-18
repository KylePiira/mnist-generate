from keras.models import Model
from keras.layers import Dense, Conv2D, Reshape, Input
from keras.utils import to_categorical
from keras.datasets import mnist
# Scipy
import numpy as np

(y_train, x_train), (y_test, x_test) = mnist.load_data()
# Desegregate training and testing data
y = np.concatenate((y_train, y_test))
x = np.concatenate((x_train, x_test))

# Make output data greyscale
y = y.astype('float32') / 255.
y = np.reshape(y, (len(y), 28, 28, 1))

# Convert labels to one-hot vectors
x = to_categorical(x)

onehot_input = Input(shape=(10,))
e = Dense(784, activation='relu')(onehot_input)
e = Reshape((28, 28, 1))(e)
e = Conv2D(16, (3, 3), activation='relu', padding='same')(e)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(e)

model = Model(onehot_input, decoded)

model.compile(loss='mse', optimizer='adam')

model.fit(x, y,
	epochs=50,
	batch_size=128,
	shuffle=True,
)

model.save('mnist_gen.model')