# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, time, copy
from chainer import cuda, Variable, optimizers, serializers, function, optimizer, initializers
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from softplus import softplus
from params import Params
import sequential
from sequential.sequential import get_weight_initializer
from sequential.link import MinibatchDiscrimination

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj

class EnergyModelParams(Params):
	def __init__(self):
		self.ndim_input = 28 * 28
		self.ndim_output = 10
		self.num_experts = 128
		self.weight_init_std = 1
		self.weight_initializer = "Normal"		# Normal or GlorotNormal or HeNormal
		self.nonlinearity = "elu"
		self.optimizer = "Adam"
		self.learning_rate = 0.001
		self.momentum = 0.5
		self.gradient_clipping = 10
		self.weight_decay = 0

class GenerativeModelParams(Params):
	def __init__(self):
		self.ndim_input = 10
		self.ndim_output = 28 * 28
		self.distribution_output = "universal"	# universal or sigmoid or tanh
		self.weight_init_std = 1
		self.weight_initializer = "Normal"		# Normal or GlorotNormal or HeNormal
		self.nonlinearity = "relu"
		self.optimizer = "Adam"
		self.learning_rate = 0.001
		self.momentum = 0.5
		self.gradient_clipping = 10
		self.weight_decay = 0

def sum_sqnorm(arr):
	sq_sum = collections.defaultdict(float)
	for x in arr:
		with cuda.get_device(x) as dev:
			x = x.ravel()
			s = x.dot(x)
			sq_sum[int(dev)] += s
	return sum([float(i) for i in six.itervalues(sq_sum)])
	
class GradientClipping(object):
	name = "GradientClipping"

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		norm = np.sqrt(sum_sqnorm([p.grad for p in opt.target.params()]))
		if norm == 0:
			return
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params():
				grad = param.grad
				with cuda.get_device(grad):
					grad *= rate

class Chain(chainer.Chain):

	def add_sequence(self, sequence, name_prefix="layer"):
		if isinstance(sequence, sequential.Sequential) == False:
			raise Exception()
		for i, link in enumerate(sequence.links):
			if isinstance(link, chainer.link.Link):
				self.add_link("{}_{}".format(name_prefix, i), link)
			elif isinstance(link, MinibatchDiscrimination):
				self.add_link("{}_{}".format(name_prefix, i), link.T)

class GAN():

	def __init__(self, params_discriminator, params_generator):
		self.params_discriminator = copy.deepcopy(params_discriminator)
		self.params_discriminator["config"] = to_object(params_discriminator["config"])

		self.params_generator = copy.deepcopy(params_generator)
		self.params_generator["config"] = to_object(params_generator["config"])

		self.build_network()
		self.setup_optimizers()
		self._gpu = False

	def get_optimizer(self, name, lr, momentum):
		if name.lower() == "adam":
			return optimizers.Adam(alpha=lr, beta1=momentum)
		if name.lower() == "adagrad":
			return optimizers.AdaGrad(lr=lr)
		if name.lower() == "adadelta":
			return optimizers.AdaDelta(rho=momentum)
		if name.lower() == "nesterov" or name.lower() == "nesterovag":
			return optimizers.NesterovAG(lr=lr, momentum=momentum)
		if name.lower() == "rmsprop":
			return optimizers.RMSprop(lr=lr, alpha=momentum)
		if name.lower() == "momentumsgd":
			return optimizers.MomentumSGD(lr=lr, mommentum=mommentum)
		if name.lower() == "sgd":
			return optimizers.SGD(lr=lr)

	def build_network(self):
		self.build_discriminator()
		self.build_generator()

	def build_discriminator(self):
		params = self.params_discriminator
		self.discriminator = Discriminator()
		self.discriminator.add_sequence(sequential.from_dict(params["discriminator"]))

	def build_generator(self):
		params = self.params_generator
		self.generator = Generator()
		self.generator.add_sequence(sequential.from_dict(params["generator"]))

	def setup_optimizers(self):
		config = self.params_discriminator["config"]
		opt = self.get_optimizer(config.optimizer, config.learning_rate, config.momentum)
		opt.setup(self.discriminator)
		if config.weight_decay > 0:
			opt.add_hook(optimizer.WeightDecay(config.weight_decay))
		if config.gradient_clipping > 0:
			opt.add_hook(GradientClipping(config.gradient_clipping))
		self.optimizer_discriminator = opt
		
		config = self.params_generator["config"]
		opt = self.get_optimizer(config.optimizer, config.learning_rate, config.momentum)
		opt.setup(self.generator)
		if config.weight_decay > 0:
			opt.add_hook(optimizer.WeightDecay(config.weight_decay))
		if config.gradient_clipping > 0:
			opt.add_hook(GradientClipping(config.gradient_clipping))
		self.optimizer_generative_model = opt

	def update_laerning_rate(self, lr):
		self.optimizer_discriminator.alpha = lr
		self.optimizer_generative_model.alpha = lr

	def to_gpu(self):
		self.discriminator.to_gpu()
		self.generator.to_gpu()
		self._gpu = True

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self._gpu

	@property
	def xp(self):
		if self.gpu_enabled:
			return cuda.cupy
		return np

	def to_variable(self, x):
		if isinstance(x, Variable) == False:
			x = Variable(x)
			if self.gpu_enabled:
				x.to_gpu()
		return x

	def to_numpy(self, x):
		if isinstance(x, Variable) == True:
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def get_batchsize(self, x):
		return x.shape[0]

	def zero_grads(self):
		self.optimizer_discriminator.zero_grads()
		self.optimizer_generative_model.zero_grads()

	def sample_z(self, batchsize=1):
		config = self.params_generator["config"]
		ndim_z = config.ndim_input
		# uniform
		z_batch = np.random.uniform(-1, 1, (batchsize, ndim_z)).astype(np.float32)
		# gaussian
		# z_batch = np.random.normal(0, 1, (batchsize, ndim_z)).astype(np.float32)
		return z_batch

	def generate_x(self, batchsize=1, test=False, as_numpy=False):
		return self.generate_x_from_z(self.sample_z(batchsize), test=test, as_numpy=as_numpy)

	def generate_x_from_z(self, z_batch, test=False, as_numpy=False):
		z_batch = self.to_variable(z_batch)
		x_batch = self.generator(z_batch, test=test)
		if as_numpy:
			return self.to_numpy(x_batch)
		return x_batch

	def backprop_discriminator(self, loss):
		self.zero_grads()
		loss.backward()
		self.optimizer_discriminator.update()

	def backprop_generator(self, loss):
		self.zero_grads()
		loss.backward()
		self.optimizer_generative_model.update()

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				filename = dir + "/{}.hdf5".format(attr)
				if os.path.isfile(filename):
					print "loading {} ...".format(filename)
					serializers.load_hdf5(filename, prop)
				else:
					print filename, "not found."

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				filename = dir + "/{}.hdf5".format(attr)
				if os.path.isfile(filename):
					os.remove(filename)
				serializers.save_hdf5(filename, prop)

class Generator(Chain):

	def add_sequence(self, sequence):
		super(Generator, self).add_sequence(sequence)
		self.sequence = sequence

	def __call__(self, z, test=False):
		return self.sequence(z, test=test)

class Discriminator(Chain):

	def add_sequence(self, sequence):
		super(Generator, self).add_sequence(sequence)
		self.sequence = sequence

	def __call__(self, x, test=False):
		return self.sequence(x, test=test)