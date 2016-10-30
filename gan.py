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

class GAN():

	def __init__(self, params_energy_model, params_generative_model):
		self.params_energy_model = copy.deepcopy(params_energy_model)
		self.params_energy_model["config"] = to_object(params_energy_model["config"])
		self.params_generative_model = copy.deepcopy(params_generative_model)
		self.params_generative_model["config"] = to_object(params_generative_model["config"])
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
		self.build_energy_model()
		self.build_generative_model()

	def build_energy_model(self):
		params = self.params_energy_model
		self.energy_model = DeepEnergyModel()
		self.energy_model.add_feature_extractor(sequential.from_dict(params["feature_extractor"]))
		self.energy_model.add_experts(sequential.from_dict(params["experts"]))
		self.energy_model.add_b(sequential.from_dict(params["b"]))

	def build_generative_model(self):
		params = self.params_generative_model
		self.generative_model = DeepGenerativeModel()
		self.generative_model.add_model(sequential.from_dict(params["model"]))

	def setup_optimizers(self):
		config = self.params_energy_model["config"]
		opt = self.get_optimizer(config.optimizer, config.learning_rate, config.momentum)
		opt.setup(self.energy_model)
		if config.weight_decay > 0:
			opt.add_hook(optimizer.WeightDecay(config.weight_decay))
		if config.gradient_clipping > 0:
			opt.add_hook(GradientClipping(config.gradient_clipping))
		self.optimizer_energy_model = opt
		
		config = self.params_generative_model["config"]
		opt = self.get_optimizer(config.optimizer, config.learning_rate, config.momentum)
		opt.setup(self.generative_model)
		if config.weight_decay > 0:
			opt.add_hook(optimizer.WeightDecay(config.weight_decay))
		if config.gradient_clipping > 0:
			opt.add_hook(GradientClipping(config.gradient_clipping))
		self.optimizer_generative_model = opt

	def update_laerning_rate(self, lr):
		self.optimizer_energy_model.alpha = lr
		self.optimizer_generative_model.alpha = lr

	def to_gpu(self):
		self.energy_model.to_gpu()
		self.generative_model.to_gpu()
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
			x.to_cpu()
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def get_batchsize(self, x):
		if isinstance(x, Variable):
			return x.data.shape[0]
		return x.shape[0]

	def zero_grads(self):
		self.optimizer_energy_model.zero_grads()
		self.optimizer_generative_model.zero_grads()

	# returns energy and product of experts
	def compute_energy(self, x_batch, test=False):
		x_batch = self.to_variable(x_batch)
		return self.energy_model(x_batch, test=test)

	def compute_energy_sum(self, x_batch, test=False):
		energy, experts = self.compute_energy(x_batch, test)
		energy = F.sum(energy) / self.get_batchsize(x_batch)
		return energy

	def compute_entropy(self):
		return self.generative_model.compute_entropy()

	def sample_z(self, batchsize=1):
		config = self.params_generative_model["config"]
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
		x_batch = self.generative_model(z_batch, test=test)
		if as_numpy:
			return self.to_numpy(x_batch)
		return x_batch

	def backprop_energy_model(self, loss):
		self.zero_grads()
		loss.backward()
		self.optimizer_energy_model.update()

	def backprop_generative_model(self, loss):
		self.zero_grads()
		loss.backward()
		self.optimizer_generative_model.update()

	def compute_kld_between_generator_and_energy_model(self, x_batch_negative):
		energy_negative, experts_negative = self.compute_energy(x_batch_negative)
		entropy = self.generative_model.compute_entropy()
		return F.sum(energy_negative) / self.get_batchsize(x_batch_negative) - entropy

	def compute_loss(self, x_batch_positive, x_batch_negative):
		energy_positive, experts_positive = self.compute_energy(x_batch_positive)
		energy_negative, experts_negative = self.compute_energy(x_batch_negative)
		energy_positive = F.sum(energy_positive) / self.get_batchsize(x_batch_positive)
		energy_negative = F.sum(energy_negative) / self.get_batchsize(x_batch_negative)
		loss = energy_positive - energy_negative
		return loss, energy_positive, energy_negative

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

class DeepGenerativeModel(Chain):

	def add_model(self, sequence):
		self.add_sequence(sequence)
		self.model = sequence

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def compute_entropy(self):
		entropy = 0
		for i, link in enumerate(self.model.links):
			if isinstance(link, L.BatchNormalization):
				entropy += F.sum(F.log(2 * math.e * math.pi * link.gamma ** 2 + 1e-8) / 2)
		return entropy

	def __call__(self, z, test=False):
		return self.model(z, test=test)

class DeepEnergyModel(Chain):

	def add_feature_extractor(self, sequence):
		self.add_sequence(sequence, "feature_extractor")
		self.feature_extractor = sequence

	def add_experts(self, sequence):
		self.add_sequence(sequence, "experts")
		self.experts = sequence

	def add_b(self, sequence):
		self.add_sequence(sequence, "b")
		self.b = sequence

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def compute_energy(self, x, features):
		experts = self.experts(features)

		# avoid overflow
		# -log(1 + exp(x)) = -max(0, x) - log(1 + exp(-|x|)) = -softplus
		product_of_experts = -softplus(experts)

		sigma = 1.0
		if x.data.ndim == 4:
			batchsize = x.data.shape[0]
			_x = F.reshape(x, (batchsize, -1))
			energy = F.sum(_x * _x, axis=1) / sigma - F.reshape(self.b(x), (-1,)) + F.sum(experts, axis=1)
		else:
			energy = F.sum(x * x, axis=1) / sigma - F.reshape(self.b(x), (-1,)) + F.sum(product_of_experts, axis=1)
		
		return energy, product_of_experts

	def __call__(self, x, test=False):
		self.test = test
		features = self.feature_extractor(x, test=test)
		energy, product_of_experts = self.compute_energy(x, features)
		return energy, product_of_experts