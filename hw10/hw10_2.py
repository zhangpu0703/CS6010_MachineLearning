import numpy as np
import matplotlib.pyplot as plt
from random import *
from math import *
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

def data_gen(thk,sep,rad,N,prob,ori_x,ori_y):
	X=[ori_x]*N
	Y=[ori_y]*N
	x_1=ori_x+thk+rad
	x_2=x_1+thk/2+rad
	y_1=ori_y+sep/2
	y_2=ori_y-sep/2
	indx=[0]*N
	data=np.zeros(shape=(N,3))
	for i in range(N):
		if random()<=0.5:
			angle=random()*pi
			X[i]=x_1+uniform(rad,rad+thk)*cos(angle)
			Y[i]=y_1+uniform(rad,rad+thk)*sin(angle)
			indx[i]=-1.0
		else:
			angle=random()*pi
			X[i]=x_2+uniform(rad,rad+thk)*cos(angle)
			Y[i]=y_2-uniform(rad,rad+thk)*sin(angle)
			indx[i]=1.0
		data[i,1]=X[i]
		data[i,2]=Y[i]
	return data,indx

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

NN_1 = KNeighborsClassifier(n_neighbors=1)
NN_3 = KNeighborsClassifier(n_neighbors=3)

data,Y=data_gen(5,5,10,2000,0.5,0,0)
X=data[:,1:3]
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
print x2_min, x2_max
xx, yy = np.meshgrid(np.arange(x1_min, x1_max,0.5),np.arange(x2_min,x2_max,0.5))

NN_1.fit(X,Y)
NN_3.fit(X,Y)
Z_1 = NN_1.predict(np.c_[xx.ravel(), yy.ravel()])
Z_3 = NN_3.predict(np.c_[xx.ravel(), yy.ravel()])
Z_1 = Z_1.reshape(xx.shape)
Z_3 = Z_3.reshape(xx.shape)

plt.figure(1)
plt.pcolormesh(xx, yy, Z_1, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("1NN classification Result")
plt.show()

plt.figure(2)
plt.pcolormesh(xx, yy, Z_3, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3NN classification Result")
plt.show()