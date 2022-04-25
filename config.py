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


# Where to find the validation dataset
if os.path.exists("/home/io18230/Desktop"): # Home Windows machine
	cfg.DATASET.RGBDCOWS2020_val2 = "/home/io18230/Desktop/RGBDCows2020_val23/Identification"
elif os.path.exists(f"/work/{cfg.GEN.BLUE_USERNAME}/Projects/ATI-workspace/datasets/RGBDCows2020_val23/Identification"): # Blue Pebble
	cfg.DATASET.RGBDCOWS2020_val2 = f"/work/{cfg.GEN.BLUE_USERNAME}/Projects/ATI-workspace/datasets/RGBDCows2020_val23/Identification"


