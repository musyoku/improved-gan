# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_device", type=int, default=0)
parser.add_argument("--image_dir", type=str, default="images")
parser.add_argument("--model_dir", type=str, default="model")
parser.add_argument("--plot_dir", type=str, default="plot")

# seed
parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()