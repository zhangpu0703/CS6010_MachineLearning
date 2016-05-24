import numpy as np
from math import *
from random import *

X = np.array([ [1,1,1] ])
y = np.array([[1]]).T
alpha,hidden_dim = 0.001,2
w_0 = 0.25*np.ones((X.shape[1],hidden_dim)) 
w_1 = 0.25*np.ones((hidden_dim+1,1)) 

def mlp_tanh(num_itm,X,y,alpha,w_0,w_1):
	for j in xrange(100):
		layer_1 = 1/(1+np.exp(-(np.dot(X,w_0))))
		nian=np.array([1]).reshape((1,1))
		layer_1=np.hstack((nian,layer_1))
		layer_2 = 1/(1+np.exp(-(np.dot(layer_1,w_1))))
		layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))
		layer_1_delta = layer_2_delta.dot(w_1.T) * (layer_1 * (1-layer_1))
		w_1 -= (alpha * layer_1.T.dot(layer_2_delta))
		w_0 -= (alpha * X.T.dot(layer_1_delta)[:,1:])
		delta_layer_1 = X.T.dot(layer_1_delta)[:,1:]
		delta_layer_2 = layer_1.T.dot(layer_2_delta)
	return w_0, w_1, delta_layer_1, delta_layer_2



def mlp_line(num_itm,X,y,alpha,w_0,w_1):
	for j in xrange(100):
		layer_1 = np.dot(X,w_0)
		nian=np.array([1]).reshape((1,1))
		layer_1=np.hstack((nian,layer_1))
		layer_2 = np.dot(layer_1,w_1)
		layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))
		layer_1_delta = layer_2_delta.dot(w_1.T) * (layer_1 * (1-layer_1))
		w_1 -= (alpha * layer_1.T.dot(layer_2_delta))
		w_0 -= (alpha * X.T.dot(layer_1_delta)[:,1:])
		delta_layer_1 = X.T.dot(layer_1_delta)[:,1:]
		delta_layer_2 = layer_1.T.dot(layer_2_delta)
	return w_0, w_1, delta_layer_1, delta_layer_2

'''
weight_0, weight_1, delta_layer_1, delta_layer_2=mlp_tanh(100,X,y,0.001,w_0,w_1)
print "The gradient in the first layer after 100  iterations:"
print delta_layer_1
print "The gradient in the second layer after 100  iterations:"
print delta_layer_2
weight_0, weight_1, delta_layer_1, delta_layer_2=mlp_line(100,X,y,0.001,w_0,w_1)
print "The gradient in the first layer after 100  iterations:"
print delta_layer_1
print "The gradient in the second layer after 100  iterations:"
print delta_layer_2
'''

w_0 = np.add(0.0001,w_0) 
w_1 = np.add(0.0001,w_1)
weight_0, weight_1, delta_layer_1, delta_layer_2=mlp_tanh(100,X,y,0.001,w_0,w_1)
print "The gradient in the first layer after 100  iterations:"
print delta_layer_1
print "The gradient in the second layer after 100  iterations:"
print delta_layer_2
weight_0, weight_1, delta_layer_1, delta_layer_2=mlp_line(100,X,y,0.001,w_0,w_1)
print "The gradient in the first layer after 100  iterations:"
print delta_layer_1
print "The gradient in the second layer after 100  iterations:"
print delta_layer_2