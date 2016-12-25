import sys, os
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import visualizer
from args import args
from model import gan

def plot(filename="gen"):
	try:
		os.mkdir(args.plot_dir)
	except:
		pass

	x_fake = gan.generate_x(100, test=True, as_numpy=True)
	x_fake = (x_fake + 1.0) / 2.0
	visualizer.tile_binary_images(x_fake.reshape((-1, 28, 28)), dir=args.plot_dir, filename=filename)

if __name__ == "__main__":
	plot()
