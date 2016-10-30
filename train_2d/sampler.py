import math, random
import numpy as np

def sample_from_gaussian_mixture(batchsize, n_dim, n_labels):
	if n_dim % 2 != 0:
		raise Exception("n_dim must be a multiple of 2.")

	def sample(x, y, label, n_labels):
		shift = 1.4
		r = 2.0 * np.pi / float(n_labels) * float(label)
		new_x = x * math.cos(r) - y * math.sin(r)
		new_y = x * math.sin(r) + y * math.cos(r)
		new_x += shift * math.cos(r)
		new_y += shift * math.sin(r)
		return np.array([new_x, new_y]).reshape((2,))

	x_var = 0.5
	y_var = 0.05
	x = np.random.normal(0, x_var, (batchsize, n_dim / 2))
	y = np.random.normal(0, y_var, (batchsize, n_dim / 2))
	z = np.empty((batchsize, n_dim), dtype=np.float32)
	for batch in xrange(batchsize):
		for zi in xrange(n_dim / 2):
			z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], random.randint(0, n_labels - 1), n_labels)

	return z

def sample_from_swiss_roll(batchsize, n_dim, n_labels):
	def sample(label, n_labels):
		uni = np.random.uniform(0.0, 3.0) / float(n_labels) + float(label) / float(n_labels)
		r = math.sqrt(uni) * 3.0
		rad = np.pi * 4.0 * math.sqrt(uni)
		x = r * math.cos(rad)
		y = r * math.sin(rad)
		return np.array([x, y]).reshape((2,))

	z = np.zeros((batchsize, n_dim), dtype=np.float32)
	for batch in xrange(batchsize):
		for zi in xrange(n_dim / 2):
			z[batch, zi*2:zi*2+2] = sample(random.randint(0, n_labels - 1), n_labels)
	
	return z