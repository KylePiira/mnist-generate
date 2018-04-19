from keras.models import load_model
# Scipy
import numpy as np
import matplotlib.pyplot as plt
# General
import argparse
import random
import time
import os

parser = argparse.ArgumentParser(description='Sample the MNIST generator.')
parser.add_argument('--num',
	default=random.randint(0,9), 
	help='the number you want to generate.',
	type=int,
)
parser.add_argument('--mode',
	default='display', 
	help='the mode to create training data in. Either: display or save',
	type=str,
)

args = parser.parse_args()

print('Generating:', args.num)

# Load the model
model = load_model('mnist_gen.model')

# Create input one-hot
x = np.zeros(10)
x[args.num] = 1.

for _ in range(10):
	# Predict vector
	prediction = model.predict(np.array([x]))[0].reshape((28,28))
	# prediction = np.round(prediction)
	prediction[prediction < 0.3] = 0

	# Pretty print prediction
	plt.imshow(prediction, cmap='Greys')

	if args.mode == 'display':
		plt.show()
	elif args.mode == 'save':
		plt.savefig('save/{}/{}.png'.format(args.num, int(time.time()*10)))