from keras.models import load_model
# Scipy
import numpy as np
import matplotlib.pyplot as plt
# General
import argparse
import random

parser = argparse.ArgumentParser(description='Sample the MNIST generator.')
parser.add_argument('--num',
	default=random.randint(0,9), 
	help='the number you want to generate.',
	type=int,
)

args = parser.parse_args()

print('Generating:', args.num)

# Load the model
model = load_model('mnist_gen.model')

# Create input one-hot
x = np.zeros(10)
x[args.num] = 1.

print(x)

# Predict vector
prediction = model.predict(np.array([x]))[0].reshape((28,28))
prediction = np.round(prediction)

# Pretty print prediction
plt.imshow(prediction, cmap='Greys')
plt.show()