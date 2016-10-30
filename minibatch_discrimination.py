import chainer
from chainer import links as L
from chainer import functions as F
from chainer import cuda

class MinibatchDiscrimination(chainer.Chain):
	def __init__(self, in_size, num_kernels, ndim_kernel=5, wscale=1, initialT=None):
		super(MinibatchDiscrimination, self).__init__(
			T=L.Linear(in_size, num_kernels * ndim_kernel, wscale=wscale, initialW=initialT)
		)

		self.num_kernels = num_kernels
		self.ndim_kernel = ndim_kernel

	def __call__(self, x):
		xp = cuda.get_array_module(x.data)
		batchsize = x.shape[0]

		M = F.reshape(self.T(x), (-1, self.num_kernels, self.ndim_kernel))
		M = F.expand_dims(M, 3)
		M_T = F.transpose(M, (3, 1, 2, 0))
		M, M_T = F.broadcast(M, M_T)

		norm = F.sum(abs(M - M_T), axis=2)
		eraser = F.broadcast_to(xp.eye(batchsize, dtype=x.dtype).reshape((batchsize, 1, batchsize)), norm.shape)
		c_b = F.exp(-(norm + 1e6 * eraser))
		o_b = F.sum(c_b, axis=2)
		return F.concat((x, o_b), axis=1)
