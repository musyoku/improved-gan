import numpy as np
import os, sys, time
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from progress import Progress
from model import discriminator_params, generator_params, gan
from args import args
from dataset import load_rgb_images
from plot import plot

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj

def sample_from_data(images, batchsize):
	example = images[0]
	height = example.shape[1]
	width = example.shape[2]
	x_batch = np.empty((batchsize, 3, height, width), dtype=np.float32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=True)
	for j in range(batchsize):
		data_index = indices[j]
		x_batch[j] = images[data_index]
	return x_batch

def main():
	images = load_rgb_images(args.image_dir)

	# config
	discriminator_config = to_object(discriminator_params["config"])
	generator_config = to_object(generator_params["config"])

	# settings
	max_epoch = 1000
	n_trains_per_epoch = 500
	batchsize_true = 128
	batchsize_fake = 128
	plot_interval = 5

	# seed
	np.random.seed(args.seed)
	if args.gpu_enabled:
		cuda.cupy.random.seed(args.seed)

	# init weightnorm layers
	if discriminator_config.use_weightnorm:
		print "initializing weight normalization layers of the discriminator ..."
		x_true = sample_from_data(images, batchsize_true)
		gan.discriminate(x_true)

	if generator_config.use_weightnorm:
		print "initializing weight normalization layers of the generator ..."
		gan.generate_x(batchsize_fake)

	# classification
	# 0 -> true sample
	# 1 -> generated sample
	class_true = gan.to_variable(np.zeros(batchsize_true, dtype=np.int32))
	class_fake = gan.to_variable(np.ones(batchsize_fake, dtype=np.int32))

	# training
	progress = Progress()
	for epoch in xrange(1, max_epoch):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_discriminator = 0
		sum_loss_generator = 0

		for t in xrange(n_trains_per_epoch):
			# sample data
			x_true = sample_from_data(images, batchsize_true)
			x_fake = gan.generate_x(batchsize_fake)

			# train discriminator
			discrimination_true, activations_true = gan.discriminate(x_true, apply_softmax=False)
			discrimination_fake, _ = gan.discriminate(x_fake, apply_softmax=False)
			loss_discriminator = F.softmax_cross_entropy(discrimination_true, class_true) + F.softmax_cross_entropy(discrimination_fake, class_fake)
			gan.backprop_discriminator(loss_discriminator)

			# train generator
			x_fake = gan.generate_x(batchsize_fake)
			discrimination_fake, activations_fake = gan.discriminate(x_fake, apply_softmax=False)
			loss_generator = F.softmax_cross_entropy(discrimination_fake, class_true)
			# feature matching
			if discriminator_config.use_feature_matching:
				features_true = activations_true[-1]
				features_fake = activations_fake[-1]
				loss_generator += F.mean_squared_error(features_true, features_fake)
			gan.backprop_generator(loss_generator)

			sum_loss_discriminator += float(loss_discriminator.data)
			sum_loss_generator += float(loss_generator.data)
			if t % 10 == 0:
				progress.show(t, n_trains_per_epoch, {})

		progress.show(n_trains_per_epoch, n_trains_per_epoch, {
			"loss (discriminator)": sum_loss_discriminator / n_trains_per_epoch,
			"loss (generator)": sum_loss_generator / n_trains_per_epoch,
		})
		gan.save(args.model_dir)

		if epoch % plot_interval == 0 or epoch == 1:
			plot(filename="epoch_{}_time_{}min".format(epoch, progress.get_total_time()))

if __name__ == "__main__":
	main()
