import numpy as np
import os, sys, time, random, math, pylab
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from progress import Progress
from model import discriminator_params, generator_params, gan
from args import args
import sampler

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj

def plot(z, color="blue", s=40):
	for n in xrange(z.shape[0]):
		result = pylab.scatter(z[n, 0], z[n, 1], s=s, marker="o", edgecolors="none", color=color)
	ax = pylab.subplot(111)
	ax.set_xlim(-4, 4)
	ax.set_ylim(-4, 4)
	pylab.xticks(pylab.arange(-4, 5))
	pylab.yticks(pylab.arange(-4, 5))

def main():
	try:
		os.mkdir(args.plot_dir)
	except:
		pass

	# config
	discriminator_config = to_object(discriminator_params["config"])
	generator_config = to_object(generator_params["config"])

	# settings
	max_epoch = 1000
	n_trains_per_epoch = 50
	batchsize_true = 100
	batchsize_fake = 100
	plotsize = 400

	fixed_z = gan.sample_z(plotsize)
	fixed_target = sampler.sample_from_swiss_roll(600, 2, 10)

	# seed
	np.random.seed(args.seed)
	if args.gpu_enabled:
		cuda.cupy.random.seed(args.seed)

	# init weightnorm layers
	if discriminator_config.use_weightnorm:
		print "initializing weight normalization layers of the discriminator ..."
		x_true = sampler.sample_from_swiss_roll(batchsize_true * 10, 2, 10)
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

			# sample from data distribution
			x_true = sampler.sample_from_swiss_roll(batchsize_true, 2, 10)
			x_fake = gan.generate_x(batchsize_fake)

			# train discriminator
			discrimination_true = gan.discriminate(x_true, apply_softmax=False)
			discrimination_fake = gan.discriminate(x_fake, apply_softmax=False)
			loss_discriminator = F.softmax_cross_entropy(discrimination_true, class_true) + F.softmax_cross_entropy(discrimination_fake, class_fake)
			gan.backprop_discriminator(loss_discriminator)

			# train generator
			x_fake = gan.generate_x(batchsize_fake)
			discrimination_fake = gan.discriminate(x_fake, apply_softmax=False)
			loss_generator = F.softmax_cross_entropy(discrimination_fake, class_true)
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

		# init
		fig = pylab.gcf()
		fig.set_size_inches(8.0, 8.0)
		pylab.clf()

		plot(fixed_target, color="#bec3c7", s=20)
		plot(gan.generate_x_from_z(fixed_z, as_numpy=True, test=True), color="#e84c3d", s=20)

		# save
		pylab.savefig("{}/{}.png".format(args.plot_dir, 100000 + epoch))

if __name__ == '__main__':
	main()
