# -*- coding: utf-8 -*-
import math
import json, os, sys, copy
from args import args
sys.path.append(os.path.split(os.getcwd())[0])
from params import Params
from gan import GAN, DiscriminatorParams, GeneratorParams
from sequential import Sequential
from sequential.layers import Linear, BatchNormalization, MinibatchDiscrimination
from sequential.functions import Activation, dropout, gaussian_noise, softmax

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

# data
mdim_data = 2
ndim_latent_code = 4

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
	config.ndim_input = mdim_data
	config.weight_init_std = 0.05
	config.weight_initializer = "Normal"
	config.use_weightnorm = False
	config.nonlinearity = "elu"
	config.optimizer = "Adam"
	config.learning_rate = 0.0002
	config.momentum = 0.5
	config.gradient_clipping = 10
	config.weight_decay = 0

	# feature extractor
	model = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	model.add(Linear(config.ndim_input, 128, use_weightnorm=config.use_weightnorm))
	model.add(Activation(config.nonlinearity))
	model.add(dropout())
	model.add(Linear(None, 128, use_weightnorm=config.use_weightnorm))
	model.add(Activation(config.nonlinearity))
	model.add(MinibatchDiscrimination(None, num_kernels=10, ndim_kernel=5))
	model.add(Linear(None, 2, use_weightnorm=config.use_weightnorm))
	# no need to add softmax() here
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
	config.ndim_output = mdim_data
	config.distribution_output = "universal"
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
	model = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	model.add(Linear(config.ndim_input, 128, use_weightnorm=config.use_weightnorm))
	model.add(BatchNormalization(128))
	model.add(Activation(config.nonlinearity))
	model.add(Linear(None, 128, use_weightnorm=config.use_weightnorm))
	model.add(BatchNormalization(128))
	model.add(Activation(config.nonlinearity))
	model.add(Linear(None, config.ndim_output, use_weightnorm=config.use_weightnorm))
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
