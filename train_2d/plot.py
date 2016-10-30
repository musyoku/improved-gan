import sys, os, random, math
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import visualizer
from args import args
from model import gan
import sampler

def main():
	try:
		os.mkdir(args.plot_dir)
	except:
		pass

	x_positive = sampler.sample_from_gaussian_mixture(1000, 2, 10)
	visualizer.plot_z(x_positive, dir=args.plot_dir, filename="positive", xticks_range=4, yticks_range=4)

	x_negative = gan.generate_x(1000, test=True)
	if params.gpu_enabled:
		x_negative.to_cpu()
	visualizer.plot_z(x_negative.data, dir=args.plot_dir, filename="negative", xticks_range=4, yticks_range=4)

if __name__ == '__main__':
	main()
