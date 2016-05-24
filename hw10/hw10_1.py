import numpy as np
import matplotlib.pyplot as plt
from random import *
from math import *
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#Training Data
X=np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
Y=[-1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0]

NN_1 = KNeighborsClassifier(n_neighbors=1)
NN_3 = KNeighborsClassifier(n_neighbors=3)

# Question A
'''
NN_1.fit(X,Y)
NN_3.fit(X,Y)
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max,0.01),np.arange(x2_min,x2_max,0.01))
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
'''
# Question B
T1=np.sqrt(np.add(np.square(X[:, 0]),np.square(X[:, 1])))
T2=np.arctan(X[:,1]-X[:,0])
TT=np.vstack((T1,T2)).T
NN_1.fit(TT,Y)
NN_3.fit(TT,Y)
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max,0.02),np.arange(x2_min,x2_max,0.02))
Z_1=np.zeros(xx.shape)
Z_3=np.zeros(xx.shape)
for i in range(xx.shape[0]):
	print i
	for j in range(xx.shape[1]):
		x_1=xx[i,j]
		x_2=yy[i,j]
		tt=np.array([sqrt(x_1**2+x_2**2),atan(x_2/x_1)])
		Z_1[i,j]=NN_1.predict(tt)
		Z_3[i,j]=NN_3.predict(tt)


plt.figure(1)
plt.pcolormesh(xx, yy, Z_1, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("1NN classification Result With Transformation")
plt.show()

plt.figure(2)
plt.pcolormesh(xx, yy, Z_3, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3NN classification Result With Transformation")
plt.show()