import sys, os
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import visualizer
from args import args
from model import params, ddgm

def main():
	try:
		os.mkdir(args.plot_dir)
	except:
		pass

	x_negative = ddgm.generate_x(100, test=True, as_numpy=True)
	x_negative = np.clip((x_negative + 1.0) / 2.0, 0, 1)
	visualizer.tile_x(x_negative, dir=args.plot_dir)

if __name__ == '__main__':
	main()
