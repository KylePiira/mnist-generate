# MNIST Generate (Keras + TensorFlow)
Generate MNIST characters from scratch!

# Using the Model
I've already pretrained a model over 15 epochs (took about 15 minutes on CPU) which is included in the Git repo.

Once you've cloned locally:
`git clone git@github.com:KylePiira/mnist-generate.git`

You should first install any missing dependencies with pip:
`pip install -r requirments.txt`

Then you are good to start generating your own unique MNIST samples using `sample.py`. The script takes one CLI parameter called `num` which should be passed a number to generate.

For example to generate a 6: `python sample.py --num 6`

If no number is specified then the script will automatically pick a random int 0-9 and draw it using Matplotlib.

# Samples
