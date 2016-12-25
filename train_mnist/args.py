# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu_device", type=int, default=0)
parser.add_argument("-m", "--model_dir", type=str, default="model")
parser.add_argument("-p", "--plot_dir", type=str, default="plot")
parser.add_argument("-l", "--num_labeled", type=int, default=100)

# seed
parser.add_argument("-s", "--seed", type=int, default=None)

args = parser.parse_args()