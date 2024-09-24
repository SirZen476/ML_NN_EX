# SGD and backpropagation are not enough to train accurate model
#based on Intro to deep learning course

import random
import torch
import numpy as np
from torch import nn, optim
import math
from IPython import display
import matplotlib.pyplot as plt
import copy
seed = 12345
random.seed(seed)
torch.manual_seed(seed)
# device check
if torch.cuda.is_available(): # use GPU if available
  device = torch.device("cuda:0")
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.enabled = False
else:
  device = torch.device("cpu")
print(device)

#data generation - 2D data samples divided to 3 diffrent classes
N = 1000  # num_samples_per_class
D = 2  # dimensions
C = 3  # num_classes

X = torch.zeros(N * C, D)
y = torch.zeros(N * C, dtype=torch.long)
for c in range(C):
    index = 0
    t = torch.linspace(0, 1, N)
    # When c = 0 and t = 0: start of linspace
    # When c = 0 and t = 1: end of linpace
    # This inner_var is for the formula inside sin() and cos() like sin(inner_var) and cos(inner_Var)
    inner_var = torch.linspace((2 * math.pi / C) * (c),(2 * math.pi / C) * (2 + c),N) + torch.randn(N) * 0.2
    #                            When t = 0                  When t = 1
    for ix in range(N * c, N * (c + 1)):
        X[ix] = t[index] * torch.FloatTensor((math.sin(inner_var[index]), math.cos(inner_var[index])))
        y[ix] = c
        index += 1

# Permute the data randomly
p = np.random.permutation(N * C)
X = X[p,:]
y = y[p]

print("Shapes:")
print("data X:", tuple(X.size()))
print("label y:", tuple(y.size()))

#generated samples, total of 3000, now to plot data points to visualize them

def cartesian_coordinate_system():
  plt.axvline(0, color='0', lw=1, zorder=0)
  plt.axhline(0, color='0', lw=1, zorder=0)
  plt.axis('off')
  plt.axis('square')

def plot_data(x,y):
  plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
  cartesian_coordinate_system()

plot_data(X,y)
plt.show()