# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
sys.path.append(os.path.split(os.getcwd())[0])
from params import Params
from gan import GAN, DiscriminatorParams, GeneratorParams, Discriminator, Generator, Sequential
from sequential.link import Linear, BatchNormalization, Deconvolution2D, Convolution2D, MinibatchDiscrimination
from sequential.function import Activation, dropout, gaussian_noise, tanh, sigmoid, reshape, reshape_1d
from sequential.util import get_conv_padding, get_paddings_of_deconv_layers, get_in_size_of_deconv_layers

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

# data
image_width = 64
image_height = image_width
ndim_latent_code = 10

# specify discriminator
discriminator_sequence_filename = args.model_dir + "/discriminator.json"

if os.path.isfile(discriminator_sequence_filename):
	print "loading", discriminator_sequence_filename
	with open(discriminator_sequence_filename, "r") as f:
		try:
			params = json.load(f)
		except Exception as e:
			raise Exception("could not load {}".format(discriminator_sequence_filename))
else:
	config = DiscriminatorParams()
	config.weight_init_std = 0.05
	config.weight_initializer = "Normal"
	config.use_weightnorm = True
	config.nonlinearity = "elu"
	config.optimizer = "Adam"
	config.learning_rate = 0.0002
	config.momentum = 0.5
	config.gradient_clipping = 10
	config.weight_decay = 0

	# feature extractor
	model = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	model.add(Convolution2D(3, 32, ksize=4, stride=2, pad=1, use_weightnorm=config.use_weightnorm))
	model.add(Activation(config.nonlinearity))
	# model.add(BatchNormalization(32))
	model.add(dropout())
	model.add(Convolution2D(32, 96, ksize=4, stride=2, pad=1, use_weightnorm=config.use_weightnorm))
	model.add(Activation(config.nonlinearity))
	# model.add(BatchNormalization(96))
	model.add(dropout())
	model.add(Convolution2D(96, 192, ksize=4, stride=2, pad=1, use_weightnorm=config.use_weightnorm))
	model.add(Activation(config.nonlinearity))
	# model.add(BatchNormalization(192))
	model.add(dropout())
	model.add(Convolution2D(192, 192, ksize=4, stride=2, pad=1, use_weightnorm=config.use_weightnorm))
	model.add(reshape_1d())
	model.add(MinibatchDiscrimination(None, num_kernels=100, ndim_kernel=5))
	model.add(Linear(None, 2, use_weightnorm=config.use_weightnorm))
	model.build()

	params = {
		"config": config.to_dict(),
		"model": model.to_dict(),
	}

	with open(discriminator_sequence_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

discriminator_params = params

# specify generator
generator_sequence_filename = args.model_dir + "/generator.json"

if os.path.isfile(generator_sequence_filename):
	print "loading", generator_sequence_filename
	with open(generator_sequence_filename, "r") as f:
		try:
			params = json.load(f)
		except:
			raise Exception("could not load {}".format(generator_sequence_filename))
else:
	config = GeneratorParams()
	config.ndim_input = ndim_latent_code
	config.distribution_output = "sigmoid"
	config.use_weightnorm = False
	config.weight_init_std = 0.05
	config.weight_initializer = "Normal"
	config.nonlinearity = "relu"
	config.optimizer = "Adam"
	config.learning_rate = 0.0002
	config.momentum = 0.5
	config.gradient_clipping = 10
	config.weight_decay = 0

	# model
	# compute projection width
	input_size = get_in_size_of_deconv_layers(image_width, num_layers=3, ksize=4, stride=2)
	# compute required paddings
	paddings = get_paddings_of_deconv_layers(image_width, num_layers=3, ksize=4, stride=2)

	model = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	model.add(Linear(config.ndim_input, 512 * input_size ** 2, use_weightnorm=config.use_weightnorm))
	model.add(Activation(config.nonlinearity))
	model.add(BatchNormalization(512 * input_size ** 2))
	model.add(reshape((-1, 512, input_size, input_size)))
	model.add(Deconvolution2D(512, 256, ksize=4, stride=2, pad=paddings.pop(0), use_weightnorm=config.use_weightnorm))
	model.add(Activation(config.nonlinearity))
	model.add(BatchNormalization(256))
	model.add(Deconvolution2D(256, 128, ksize=4, stride=2, pad=paddings.pop(0), use_weightnorm=config.use_weightnorm))
	model.add(Activation(config.nonlinearity))
	model.add(BatchNormalization(128))
	model.add(Deconvolution2D(128, 3, ksize=4, stride=2, pad=paddings.pop(0), use_weightnorm=config.use_weightnorm))
	if config.distribution_output == "sigmoid":
		model.add(sigmoid())
	if config.distribution_output == "tanh":
		model.add(tanh())
	model.build()

	params = {
		"config": config.to_dict(),
		"model": model.to_dict(),
	}

	with open(generator_sequence_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

generator_params = params

gan = GAN(discriminator_params, generator_params)
gan.load(args.model_dir)

if args.gpu_enabled == 1:
	gan.to_gpu()
