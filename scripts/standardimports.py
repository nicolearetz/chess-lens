import sys  # for importing from a different directory
sys.path.insert(0, '../source')
sys.path.insert(0, '../images')

import imageio as iio
import matplotlib.pyplot as plt
import cv2

import numpy as np
import scipy.linalg as la

import time
tStart_notebook = time.time()