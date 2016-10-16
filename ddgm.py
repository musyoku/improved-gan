# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, time
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from softplus import softplus

activations = {
	"sigmoid": F.sigmoid, 
	"tanh": F.tanh, 
	"softplus": F.softplus, 
	"relu": F.relu, 
	"leaky_relu": F.leaky_relu, 
	"elu": F.elu
}

class Params():
	def __init__(self, dict=None):
		self.ndim_x = 28 * 28
		self.ndim_z = 10
		self.apply_dropout = False
		self.distribution_x = "universal"	# universal or sigmoid or tanh

		self.energy_model_num_experts = 128
		self.energy_model_feature_extractor_hidden_units = [500]
		self.energy_model_batchnorm_to_input = True
		# True:  y = f(BN(W*x + b))
		# False: y = f(W*BN(x) + b))
		self.energy_model_batchnorm_before_activation = False
		self.energy_model_batchnorm_enabled = True
		self.energy_model_wscale = 1
		self.energy_model_activation_function = "elu"
		self.energy_model_optimizer = "Adam"
		self.energy_model_learning_rate = 0.001
		self.energy_model_momentum = 0.9
		self.energy_model_gradient_clipping = 10
		self.energy_model_weight_decay = 0

		self.generative_model_hidden_units = [500]
		self.generative_model_batchnorm_to_input = False
		self.generative_model_batchnorm_before_activation = False
		self.generative_model_batchnorm_enabled = True
		self.generative_model_wscale = 1
		self.generative_model_activation_function = "elu"
		self.generative_model_optimizer = "Adam"
		self.generative_model_learning_rate = 0.001
		self.generative_model_momentum = 0.9
		self.generative_model_gradient_clipping = 10
		self.generative_model_weight_decay = 0

		self.gpu_enabled = True

		if dict:
			self.from_dict(dict)
			self.check()

	def from_dict(self, dict):
		for attr, value in dict.iteritems():
			if hasattr(self, attr):
				setattr(self, attr, value)

	def to_dict(self):
		dict = {}
		for attr, value in self.__dict__.iteritems():
			dict[attr] = value
		return dict

	def dump(self):
		print "params:"
		for attr, value in self.__dict__.iteritems():
			print "	{}: {}".format(attr, value)

	def check(self):
		base = Params()
		for attr, value in self.__dict__.iteritems():
			if not hasattr(base, attr):
				raise Exception("invalid parameter '{}'".format(attr))

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

class DDGM():

	def __init__(self, params):
		params.check()
		self.params = params
		self.create_network()
		self.setup_optimizers()

	def create_network(self):
		params = self.params

		# deep energy model
		attributes = {}
		units = [(params.ndim_x, params.energy_model_feature_extractor_hidden_units[0])]
		units += zip(params.energy_model_feature_extractor_hidden_units[:-1], params.energy_model_feature_extractor_hidden_units[1:])
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=params.energy_model_wscale)
			if params.energy_model_batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		attributes["b"] = L.Linear(params.ndim_x, 1, wscale=params.energy_model_wscale, nobias=True)
		attributes["feature_detector"] = L.Linear(params.energy_model_feature_extractor_hidden_units[-1], params.energy_model_num_experts, wscale=params.energy_model_wscale)
		
		self.energy_model = DeepEnergyModel(params, n_layers=len(units), **attributes)

		# deep generative model
		attributes = {}
		units = [(params.ndim_z, params.generative_model_hidden_units[0])]
		units += zip(params.generative_model_hidden_units[:-1], params.generative_model_hidden_units[1:])
		units += [(params.generative_model_hidden_units[-1], params.ndim_x)]
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=params.generative_model_wscale)
			if params.generative_model_batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		if params.distribution_x == "sigmoid":
			self.generative_model = SigmoidDeepGenerativeModel(params, n_layers=len(units), **attributes)
		elif params.distribution_x == "tanh":
			self.generative_model = TanhDeepGenerativeModel(params, n_layers=len(units), **attributes)
		elif params.distribution_x == "universal":
			self.generative_model = DeepGenerativeModel(params, n_layers=len(units), **attributes)
		else:
			raise Exception()

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

	def setup_optimizers(self):
		params = self.params
		
		opt = self.get_optimizer(params.energy_model_optimizer, params.energy_model_learning_rate, params.energy_model_momentum)
		opt.setup(self.energy_model)
		if params.energy_model_weight_decay > 0:
			opt.add_hook(optimizer.WeightDecay(params.energy_model_weight_decay))
		if params.energy_model_gradient_clipping > 0:
			opt.add_hook(GradientClipping(params.energy_model_gradient_clipping))
		self.optimizer_energy_model = opt
		
		opt = self.get_optimizer(params.generative_model_optimizer, params.generative_model_learning_rate, params.generative_model_momentum)
		opt.setup(self.generative_model)
		if params.generative_model_weight_decay > 0:
			opt.add_hook(optimizer.WeightDecay(params.generative_model_weight_decay))
		if params.generative_model_gradient_clipping > 0:
			opt.add_hook(GradientClipping(params.generative_model_gradient_clipping))
		self.optimizer_generative_model = opt

	def update_laerning_rate(self, lr):
		self.optimizer_energy_model.alpha = lr
		self.optimizer_generative_model.alpha = lr

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self.params.gpu_enabled

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

	def compute_energy(self, x_batch, test=False):
		x_batch = self.to_variable(x_batch)
		return self.energy_model(x_batch, test=test)

	def compute_entropy(self):
		return self.generative_model.compute_entropy()

	def sample_z(self, batchsize=1):
		# uniform
		z_batch = np.random.uniform(-1, 1, (batchsize, self.params.ndim_z)).astype(np.float32)
		# gaussian
		# z_batch = np.random.normal(0, 1, (batchsize, self.params.ndim_z)).astype(np.float32)
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
					print "loading",  filename
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
		print "model saved."

class DeepGenerativeModel(chainer.Chain):
	def __init__(self, params, n_layers, **layers):
		super(DeepGenerativeModel, self).__init__(**layers)

		self.n_layers = n_layers
		self.activation_function = params.generative_model_activation_function
		self.apply_dropout = params.apply_dropout
		self.batchnorm_enabled = params.generative_model_batchnorm_enabled
		self.batchnorm_to_input = params.generative_model_batchnorm_to_input
		self.batchnorm_before_activation = params.generative_model_batchnorm_before_activation

		if params.gpu_enabled:
			self.to_gpu()

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def compute_entropy(self):
		entropy = 0

		if self.batchnorm_enabled == False:
			return entropy

		for i in range(self.n_layers):
			bn = getattr(self, "batchnorm_%d" % i)
			entropy += F.sum(F.log(2 * math.e * math.pi * bn.gamma ** 2 + 1e-8) / 2)

		return entropy

	def compute_output(self, z):
		f = activations[self.activation_function]
		chain = [z]

		for i in range(self.n_layers):
			u = chain[-1]

			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)

			if self.batchnorm_enabled:
				bn = getattr(self, "batchnorm_%d" % i)
				if i == 0:
					if self.batchnorm_to_input == True:
						u = bn(u, test=self.test)
				elif i == self.n_layers - 1:
					if self.batchnorm_before_activation == False:
						u = bn(u, test=self.test)
				else:
					u = bn(u, test=self.test)

			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)

			if i == self.n_layers - 1:
				output = u
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not self.test)

			chain.append(output)

		return chain[-1]

	def __call__(self, z, test=False):
		self.test = test
		return self.compute_output(z)

class SigmoidDeepGenerativeModel(DeepGenerativeModel):
	def __call__(self, z, test=False):
		self.test = test
		return F.sigmoid(self.compute_output(z))

class TanhDeepGenerativeModel(DeepGenerativeModel):
	def __call__(self, z, test=False):
		self.test = test
		return F.tanh(self.compute_output(z))

class DeepEnergyModel(chainer.Chain):
	def __init__(self, params, n_layers, **layers):
		super(DeepEnergyModel, self).__init__(**layers)

		self.n_layers = n_layers
		self.activation_function = params.energy_model_activation_function
		self.apply_dropout = params.apply_dropout
		self.batchnorm_enabled = params.energy_model_batchnorm_enabled
		self.batchnorm_to_input = params.energy_model_batchnorm_to_input
		self.batchnorm_before_activation = params.energy_model_batchnorm_before_activation

		if params.gpu_enabled:
			self.to_gpu()

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def extract_features(self, x):
		f = activations[self.activation_function]
		chain = [x]

		for i in range(self.n_layers):
			u = chain[-1]

			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)

			if self.batchnorm_enabled:
				bn = getattr(self, "batchnorm_%d" % i)
				if i == 0:
					if self.batchnorm_to_input == True:
						u = bn(u, test=self.test)
				else:
					u = bn(u, test=self.test)

			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)

			if i == self.n_layers - 1:
				output = F.tanh(u)
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not self.test)

			chain.append(output)

		return chain[-1]

	def compute_energy(self, x, features):
		feature_detector = self.feature_detector(features)

		# avoid overflow
		# -log(1 + exp(x)) = -max(0, x) - log(1 + exp(-|x|)) = -softplus
		experts = -softplus(feature_detector)

		sigma = 1.0
		energy = F.sum(x * x, axis=1) / sigma - F.reshape(self.b(x), (-1,)) + F.sum(experts, axis=1)
		
		return energy, experts

	def __call__(self, x, test=False):
		self.test = test
		features = self.extract_features(x)
		energy, experts = self.compute_energy(x, features)
		return energy, experts