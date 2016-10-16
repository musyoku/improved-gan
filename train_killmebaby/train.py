import numpy as np
import os, sys, time
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from model import params, dcdgm
from args import args
from dataset import load_rgb_images

def sample_from_data(images, batchsize):
	example = images[0]
	height = example.shape[1]
	width = example.shape[2]
	x_batch = np.empty((batchsize, 3, height, width), dtype=np.float32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=True)
	for j in range(batchsize):
		data_index = indices[j]
		x_batch[j] = (images[data_index] - 0.5) * 2
	return x_batch

def main():
	# load MNIST images
	images = load_rgb_images(args.image_dir)
	
	# settings
	max_epoch = 1000
	n_trains_per_epoch = 100
	batchsize_positive = 128
	batchsize_negative = 128

	# seed
	np.random.seed(args.seed)
	if params.gpu_enabled:
	    cuda.cupy.random.seed(args.seed)

	total_time = 0
	for epoch in xrange(1, max_epoch):
		sum_energy_positive = 0
		sum_energy_negative = 0
		sum_kld = 0
		epoch_time = time.time()

		for t in xrange(n_trains_per_epoch):
			# sample from data distribution
			x_positive = sample_from_data(images, batchsize_positive)

			# sample from generator
			x_negative = dcdgm.generate_x(batchsize_negative)

			if True:
				loss, energy_positive, energy_negative = dcdgm.compute_loss(x_positive, x_negative)
				dcdgm.backprop_energy_model(loss)
			else:
				energy_positive, experts_positive = dcdgm.compute_energy(x_positive)
				energy_positive = F.sum(energy_positive) / dcdgm.get_batchsize(x_positive)
				dcdgm.backprop_energy_model(energy_positive)
				
				energy_negative, experts_negative = dcdgm.compute_energy(x_negative)
				energy_negative = F.sum(energy_negative) / dcdgm.get_batchsize(x_negative)
				dcdgm.backprop_energy_model(-energy_negative)

			# train generative model
			# TODO: KLD must be greater than or equal to 0
			x_negative = dcdgm.generate_x(batchsize_negative)
			kld = dcdgm.compute_kld_between_generator_and_energy_model(x_negative)
			dcdgm.backprop_generative_model(kld)

			sum_energy_positive += float(energy_positive.data)
			sum_energy_negative += float(energy_negative.data)
			sum_kld += float(kld.data)
			if t % 10 == 0:
				sys.stdout.write("\rTraining in progress...({} / {})".format(t, n_trains_per_epoch))
				sys.stdout.flush()

		epoch_time = time.time() - epoch_time
		total_time += epoch_time
		sys.stdout.write("\r")
		print "epoch: {} energy: x+ {:.3f} x- {:.3f} kld: {:.3f} time: {} min total: {} min".format(epoch + 1, sum_energy_positive / n_trains_per_epoch, sum_energy_negative / n_trains_per_epoch, sum_kld / n_trains_per_epoch, int(epoch_time / 60), int(total_time / 60))
		sys.stdout.flush()
		dcdgm.save(args.model_dir)

if __name__ == '__main__':
	main()
