#!/usr/bin/env python

# Core libraries
import os
from easydict import EasyDict as edict

"""
This file is for managing ALL constants and configuration values
They're all nested per python class

To import the file:
from config import cfg

easydict usage:
https://pypi.python.org/pypi/easydict/
"""

# The main configuration variable / dictionary
cfg = edict()

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

"""
Generic constants & parameters
"""

cfg.GEN = edict()

# User name on Blue Pebble/Crystal (CHANGE THIS TO YOUR OWN USERNAME)
cfg.GEN.BLUE_USERNAME = "io18230"

# The base directory for this repository (automatically set based on available directories)
if os.path.exists("/home/io18230/Desktop"): # Home windows machine
	cfg.GEN.BASE_DIR = "/home/io18230/Desktop/temp/code/ATI-Pilot-Project-masterfriday/ATI-Pilot-Project-master"
if os.path.exists(f"/home/{cfg.GEN.BLUE_USERNAME}/Projects/ATI-Pilot-Project"): # Blue pebble/crystal
	cfg.GEN.BASE_DIR = f"/home/{cfg.GEN.BLUE_USERNAME}/Projects/ATI-Pilot-Project"
else: #Jing's GPU computer
	cfg.GEN.BASE_DIR = "/home/will/Desktop/io18230/Projects/ATI-Pilot-Project"





"""
Dataset constants
"""

# The base dictionary
cfg.DATASET = edict()

# Where to find the RGBDCows2020 dataset
if os.path.exists("/home/io18230/Desktop"): # Home Windows machine
	cfg.DATASET.RGBDCOWS2020_LOC = "/home/io18230/Desktop/RGBDCows2020/ideal"
elif os.path.exists(f"/work/{cfg.GEN.BLUE_USERNAME}/Projects/ATI-workspace/datasets/RGBDCows2020/ideal"): # Blue Pebble
	cfg.DATASET.RGBDCOWS2020_LOC = f"/work/{cfg.GEN.BLUE_USERNAME}/Projects/ATI-workspace/datasets/RGBDCows2020/ideal"
else:
	cfg.DATASET.RGBDCOWS2020_LOC = f"/home/will/Desktop/io18230/Projects/ATI-workspace/datasets/RGBDCows2020/ideal"

# Where to find the validation dataset
if os.path.exists("/home/io18230/Desktop"): # Home Windows machine
	cfg.DATASET.RGBDCOWS2020_val2 = "/home/io18230/Desktop/RGBDCows2020_val23/Identification"
elif os.path.exists(f"/work/{cfg.GEN.BLUE_USERNAME}/Projects/ATI-workspace/datasets/RGBDCows2020_val23/Identification"): # Blue Pebble
	cfg.DATASET.RGBDCOWS2020_val2 = f"/work/{cfg.GEN.BLUE_USERNAME}/Projects/ATI-workspace/datasets/RGBDCows2020_val23/Identification"
else:
	cfg.DATASET.RGBDCOWS2020_val2 = f"/home/will/Desktop/io18230/Projects/ATI-workspace/datasets/RGBDCows2020_val23/Identification"

# Where to find the OpenSetCows2020 dataset
if os.path.exists("/home/will"): # Linux machine
	cfg.DATASET.OPENSETCOWS2020_LOC = "/home/will/work/1-RA/src/Datasets/data/OpenSetCows2019"
elif os.path.exists(f"/home/{cfg.GEN.BLUE_USERNAME}"): # Blue pebble
	cfg.DATASET.OPENSETCOWS2020_LOC = f"/work/{cfg.GEN.BLUE_USERNAME}/datasets/OpenCows2020/identification"
else:
	cfg.DATASET.OPENSETCOWS2020_LOC = f"/home/will/Desktop/io18230/Projects/ATI-Pilot-Project/src/Datasets/data/OpenSetCows2019"

