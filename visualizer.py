import pylab
import numpy as np
from StringIO import StringIO
from PIL import Image

def tile_binary_images(x, dir=None, filename="x", row=10, col=10):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(col * 2, row * 2)
	pylab.clf()
	pylab.gray()
	for m in range(row * col):
		pylab.subplot(row, col, m + 1)
		pylab.imshow(np.clip(x[m], 0, 1), interpolation="none")
		pylab.axis("off")
	pylab.savefig("{}/{}.png".format(dir, filename))

def tile_rgb_images(x, dir=None, filename="x", row=10, col=10):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(col * 2, row * 2)
	pylab.clf()
	for m in range(row * col):
		pylab.subplot(row, col, m + 1)
		pylab.imshow(np.clip(x[m], 0, 1), interpolation="none")
		pylab.axis("off")
	pylab.savefig("{}/{}.png".format(dir, filename))

def plot_z(z, dir=None, filename="z", xticks_range=None, yticks_range=None):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	for n in xrange(z.shape[0]):
		result = pylab.scatter(z[n, 0], z[n, 1], s=40, marker="o", edgecolors='none')
	pylab.xlabel("z1")
	pylab.ylabel("z2")
	if xticks_range is not None:
		pylab.xticks(pylab.arange(-xticks_range, xticks_range + 1))
	if yticks_range is not None:
		pylab.yticks(pylab.arange(-yticks_range, yticks_range + 1))
	pylab.savefig("{}/{}.png".format(dir, filename))

