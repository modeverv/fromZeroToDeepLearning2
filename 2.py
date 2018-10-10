import sys
import os
libr = os.path.dirname(os.path.abspath(__file__)) + '/deep-learning-from-scratch-2'
sys.path.append(libr)
import numpy as np
from common.layers import MatMul

c = np.array([1, 0, 0, 0, 0, 0, 0])
W = np.random.randn(7, 3)
layer = MatMul(W)
h = layer.forward(c)
print(h)

c0 = np.array([1, 0, 0, 0, 0, 0, 0])
c1 = np.array([0, 0, 1, 0, 0, 0, 0])
W_in = np.random.randn(7, 3)
W_out = np.random.randon(3, 7)

in_layer0 = MutMal(W_in)


