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

	x_negative = gan.generate_x(100, test=True, as_numpy=True)
	visualizer.tile_rgb_images(x_negative.transpose(0, 2, 3, 1), dir=args.plot_dir, filename=filename)

if __name__ == '__main__':
	plot()
