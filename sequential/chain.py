import os
import chainer
import sequential
from chainer import optimizers, serializers

class Chain(chainer.Chain):

	def add_sequence(self, sequence):
		if isinstance(sequence, sequential.Sequential) == False:
			raise Exception()
		for i, link in enumerate(sequence.links):
			if isinstance(link, chainer.link.Link):
				self.add_link("link_{}".format(i), link)
		self.sequence = sequence

	def load(self, filename):
		if os.path.isfile(filename):
			print "loading {} ...".format(filename)
			serializers.load_hdf5(filename, self)
		else:
			print filename, "not found."

	def save(self, filename):
		if os.path.isfile(filename):
			os.remove(filename)
		serializers.save_hdf5(filename, self)

	def get_optimizer(self, name, lr, momentum=0.9):
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

	def setup_optimizers(self, optimizer_name, lr, momentum=0.9, weight_decay=0, gradient_clipping=0):
		opt = self.get_optimizer(optimizer_name, lr, momentum)
		opt.setup(self)
		if weight_decay > 0:
			opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))
		if gradient_clipping > 0:
			opt.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))
		self.optimizer = opt

	def backprop(self, loss):
		self.optimizer.zero_grads()
		loss.backward()
		self.optimizer.update()

	def __call__(self, x, test=False):
		return self.sequence(x, test=test)
