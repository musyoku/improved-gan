# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu-device", type=int, default=0)
parser.add_argument("-m", "--model-dir", type=str, default="model")
parser.add_argument("-p", "--plot-dir", type=str, default="plot")
parser.add_argument("-i", "--image-dir", type=str, default="images")

# seed
parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()