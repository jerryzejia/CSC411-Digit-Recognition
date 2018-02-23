from scipy import *
from scipy.io import loadmat
import part2
import os
import numpy as np

def gradient(x, y, w, b):
    p = part2.f(x, w, b)
    return dot(x, (y - p).T)


