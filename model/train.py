from model import CNN_model
import numpy as np

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_val = np.load('x_val.npy')
y_val = np.load('y_val.npy')
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
print(y_train)
model = CNN_model(x_train, y_train, x_val, y_val)
model.train()
