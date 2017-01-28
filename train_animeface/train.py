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
	discriminator_config = gan.config_discriminator
	generator_config = gan.config_generator

	# settings
	max_epoch = 1000
	num_updates_per_epoch = 500
	batchsize_true = 128
	batchsize_fake = 128
	plot_interval = 5

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# init weightnorm layers
	if discriminator_config.use_weightnorm:
		print "initializing weight normalization layers of the discriminator ..."
		x_true = sample_from_data(images, batchsize_true)
		gan.discriminate(x_true)

	if generator_config.use_weightnorm:
		print "initializing weight normalization layers of the generator ..."
		gan.generate_x(batchsize_fake)

	# training
	progress = Progress()
	for epoch in xrange(1, max_epoch + 1):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_unsupervised = 0
		sum_loss_adversarial = 0
		sum_dx_unlabeled = 0
		sum_dx_generated = 0

		for t in xrange(num_updates_per_epoch):
			# sample data
			x_true = sample_from_data(images, batchsize_true)
			x_fake = gan.generate_x(batchsize_fake)
			x_fake.unchain_backward()

			# unsupervised loss
			# D(x) = Z(x) / {Z(x) + 1}, where Z(x) = \sum_{k=1}^K exp(l_k(x))
			# softplus(x) := log(1 + exp(x))
			# logD(x) = logZ(x) - log(Z(x) + 1)
			# 		  = logZ(x) - log(exp(log(Z(x))) + 1)
			# 		  = logZ(x) - softplus(logZ(x))
			# 1 - D(x) = 1 / {Z(x) + 1}
			# log{1 - D(x)} = log1 - log(Z(x) + 1)
			# 				= -log(exp(log(Z(x))) + 1)
			# 				= -softplus(logZ(x))
			log_zx_u, activations_u = gan.discriminate(x_true, apply_softmax=False)
			log_dx_u = log_zx_u - F.softplus(log_zx_u)
			dx_u = F.sum(F.exp(log_dx_u)) / batchsize_true
			loss_unsupervised = -F.sum(log_dx_u) / batchsize_true	# minimize negative logD(x)
			py_x_g, _ = gan.discriminate(x_fake, apply_softmax=False)
			log_zx_g = F.logsumexp(py_x_g, axis=1)
			loss_unsupervised += F.sum(F.softplus(log_zx_g)) / batchsize_true	# minimize negative log{1 - D(x)}

			# update discriminator
			gan.backprop_discriminator(loss_unsupervised)

			sum_loss_unsupervised += float(loss_unsupervised.data)
			sum_dx_unlabeled += float(dx_u.data)

			# generator loss
			x_fake = gan.generate_x(batchsize_fake)
			log_zx_g, activations_g = gan.discriminate(x_fake, apply_softmax=False)
			log_dx_g = log_zx_g - F.softplus(log_zx_g)
			dx_g = F.sum(F.exp(log_dx_g)) / batchsize_fake
			loss_generator = -F.sum(log_dx_g) / batchsize_true	# minimize negative logD(x)

			# feature matching
			if discriminator_config.use_feature_matching:
				features_true = activations_u[-1]
				features_true.unchain_backward()
				if batchsize_true != batchsize_fake:
					x_fake = gan.generate_x(batchsize_true)
					_, activations_g = gan.discriminate(x_fake, apply_softmax=False)
				features_fake = activations_g[-1]
				loss_generator += F.mean_squared_error(features_true, features_fake)

			# update generator
			gan.backprop_generator(loss_generator)

			sum_loss_adversarial += float(loss_generator.data)
			sum_dx_generated += float(dx_g.data)
			if t % 10 == 0:
				progress.show(t, num_updates_per_epoch, {})

		gan.save(args.model_dir)

		progress.show(num_updates_per_epoch, num_updates_per_epoch, {
			"loss_u": sum_loss_unsupervised / num_updates_per_epoch,
			"loss_g": sum_loss_adversarial / num_updates_per_epoch,
			"dx_u": sum_dx_unlabeled / num_updates_per_epoch,
			"dx_g": sum_dx_generated / num_updates_per_epoch,
		})

		if epoch % plot_interval == 0 or epoch == 1:
			plot(filename="epoch_{}_time_{}min".format(epoch, progress.get_total_time()))

if __name__ == "__main__":
	main()
