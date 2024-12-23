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
#partition data to train and val:

TrainInput,TrainLabel,ValidInput,ValidLabel = X[:2500,:],y[:2500],X[2501:,:],y[2501:]
# model 1 to train linear model 2x3 affine mapping as 2x100 to 100x3 mapping

H= 100 # num of hidden units
Linear_model = nn.Sequential(
    nn.Linear(D,H),# 2x100
    nn.Linear(H,C)# 100x3
)
#display model shape
print(Linear_model)
print(Linear_model[0])
print(Linear_model[0].weight.shape)
print(Linear_model[0].weight.requires_grad)
print(Linear_model[0].weight.grad)

# display  weights:

print(type(Linear_model.state_dict()))
for k, v in Linear_model.state_dict().items():
  print(k, v.shape)

Linear_model_initial_weights = copy.deepcopy(Linear_model.state_dict())
print(Linear_model_initial_weights)


# next model Neural Network , added Relu layer in middle

NN_model = nn.Sequential(
    nn.Linear(D,H),# 2x100
    nn.ReLU(),# activation function
    nn.Linear(H,C)# 100x3
)
# print model and  weights
print(NN_model)
NN_model_initial_weights = copy.deepcopy(NN_model.state_dict())
print(NN_model_initial_weights)

#display init weights

print(Linear_model[0].bias[:10])
print(NN_model[0].bias[:10])

criterion = torch.nn.CrossEntropyLoss()
minibatch_size = 500
n_epochs = 1000


def train_model(model,optimizer):
    train_acc = []
    valid_acc = []
    #Training
    for t in range(n_epochs):
        #premutate the data and divide to minibatch
        p = np.random.permutation(len(TrainInput))
        train_data = TrainInput[p]
        train_label = TrainLabel[p]
        acc, Validacc = 0.0, 0.0
        for i in range(0, train_data.shape[0], minibatch_size):
            pred = model(train_data[i:i+minibatch_size])# calc pred
            loss = criterion(pred,train_label[i:i+minibatch_size])# calc loss
            optimizer.zero_grad()#zero grad of optimizer
            loss.backward()# backprop
            optimizer.step()#forward path
            # Compute training accuracy
        #compare training accuracy
        score, predicted = torch.max(model(train_data), 1)
        acc = (train_label == predicted).sum().float() / len(train_label)
        #compar val acc
        _, predicted = torch.max(model(ValidInput), 1)
        Validacc = (ValidLabel == predicted).sum().float() / len(ValidLabel)
        '''print("[EPOCH]: %i, [LOSS]: %.6f, [TRAIN ACCURACY]: %.3f, [VALID ACCURACY]: %.3f" % (
        t, loss.item(), acc,Validacc))''' # to print every epoch
        if (t== n_epochs-1 ):# for last print - to see change
            print("[EPOCH]: %i, [LOSS]: %.6f, [TRAIN ACCURACY]: %.3f, [VALID ACCURACY]: %.3f" % (
                t, loss.item(), acc, Validacc))
        display.clear_output(wait=True)
        # Save error on each epoch
        train_acc.append(acc)
        valid_acc.append(Validacc)

    return train_acc, valid_acc

# we use the optim package to apply SGD for our parameter updates
learning_rate = 1e-03
Linear_model_optimizer = torch.optim.SGD(Linear_model.parameters(), lr=learning_rate)
NN_model_optimizer = torch.optim.SGD(NN_model.parameters(), lr=learning_rate)
# training Linear model
print("Linear Model : \n")
train_acc_Linear_model, valid_acc_Linear_model = train_model(Linear_model, Linear_model_optimizer)# val acc at end is 0.519
# training NN
print("NN Model : \n")
train_acc_NN_model, valid_acc_NN_model = train_model(NN_model, NN_model_optimizer)# vall acc is 0.527 - not oprimized

#first - optimize the learning rate
learning_rate = 0.05
Linear_model_optimizer.param_groups[0]['lr'] = learning_rate
NN_model_optimizer.param_groups[0]['lr'] = learning_rate

Linear_model.load_state_dict(Linear_model_initial_weights)
NN_model.load_state_dict(NN_model_initial_weights)
# training Linear model
print("Linear Model(opt learning rate) : \n")
train_acc_Linear_model, valid_acc_Linear_model = train_model(Linear_model, Linear_model_optimizer)# val acc at end is 0.515 - no big change from learning rate
# training NN
print("NN Model(opt learning rate) : \n")
train_acc_NN_model, valid_acc_NN_model = train_model(NN_model, NN_model_optimizer)# vall acc is 0.915 - huge improvment

# optimizer selection - we will use adams optimizer
Linear_model_optimizer = torch.optim.Adam(Linear_model.parameters())
NN_model_optimizer = torch.optim.Adam(NN_model.parameters())

Linear_model.load_state_dict(Linear_model_initial_weights)
NN_model.load_state_dict(NN_model_initial_weights)

n_epochs = 500# decrease epochs for training

print("Linear Model (adam): \n")
train_acc_Linear_model, valid_acc_Linear_model = train_model(Linear_model, Linear_model_optimizer)# val acc at end is 0.519 - no big change from learning rate or adam
# training NN
print("NN Model : (adam) \n")
train_acc_NN_model, valid_acc_NN_model = train_model(NN_model, NN_model_optimizer)# vall acc is 0.994 - more improvmnent, less epochs

# plot of learning curves
def plot_curves(train, valid, name):
  plt.title(f'Training Curve - {name}')
  plt.plot(range(len(train)), train, label="Train")
  plt.plot(range(len(valid)), valid, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend(loc='best')
  plt.show()

plot_curves(train_acc_Linear_model, valid_acc_Linear_model, "Linear")
plot_curves(train_acc_NN_model, valid_acc_NN_model, "Neural Network ")
plt.show()

# we can see that NN performance is much better - linear training curve is not good
# now to plot model pred with data
def plot_model(X, y, model):
    mesh = np.arange(-1.1, 1.1, 0.01)
    xx, yy = np.meshgrid(mesh, mesh)
    with torch.no_grad():
        data = torch.from_numpy(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float()
        Z = model(data).detach()
    Z = np.argmax(Z, axis=1).reshape(xx.shape)


    plt.subplot(1,3,1)
    plot_data(X, y)

    plt.subplot(1,3,2)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
    cartesian_coordinate_system()

    plt.subplot(1,3,3)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
    plot_data(X, y)
    plt.show()

# plot trained model
plt.title("Linear model predictions:")
plot_model(X, y, Linear_model)
plt.title("NN model predictions:")
plot_model(X, y, NN_model)

