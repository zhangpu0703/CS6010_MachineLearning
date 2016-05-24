import numpy as np
import matplotlib.pyplot as plt
from random import *
from math import *
from sklearn import linear_model
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
import csv

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

train=np.genfromtxt("train.csv",dtype=None,delimiter=',')
test=np.genfromtxt("test.csv",dtype=None,delimiter=',')
train_x=np.array(train[1:,1:3], dtype='|S4')
train_x = train_x.astype(np.float)
train_y=np.array(train[1:,0], dtype='|S4')
train_y = train_y.astype(np.float)
test_x=np.array(test[1:,1:3], dtype='|S4')
test_x = test_x.astype(np.float)
test_y=np.array(test[1:,0], dtype='|S4')
test_y = test_y.astype(np.float)
'''
# Question A
size_K=100
cv_error=[0]*size_K
for k in range(1,size_K+1):
	knn= KNeighborsClassifier(n_neighbors=k)
	all_list=range(len(train_x))
	n_folds=10
	valid_error=[0]*n_folds
	for fold in range(n_folds):
		valid_idx=range(fold*(len(train_x)/n_folds),(fold+1)*(len(train_x)/n_folds))
		train_idx=list(set(all_list)-set(valid_idx))
		cv_train_x=train_x[train_idx]
		cv_train_y=train_y[train_idx]
		cv_valid_x=train_x[valid_idx]
		cv_valid_y=train_y[valid_idx]
		knn.fit(cv_train_x,cv_train_y)
		pred_valid=knn.predict(cv_valid_x)
		valid_error[fold]=(sum(pred_valid!=cv_valid_y)+0.0)/len(pred_valid)

	cv_error[k-1]=sum(valid_error)/len(valid_error)

#print accuracy
plt.figure()
plt.plot(range(1,size_K+1),cv_error,'b.-')
plt.title("Choose K to Minimize CV Error")
plt.xlabel("Values of K")
plt.ylabel("CV Errors")
plt.show()

opt_K=np.argmin(cv_error)+1
print "The Best Choice of K to minimize CV error is:", opt_K
'''
# Question B

knn= KNeighborsClassifier(n_neighbors=13)
all_list=range(len(train_y))
n_folds=10
valid_error=[0]*n_folds
for fold in range(n_folds):
	valid_idx=range(fold*(len(train_y)/n_folds),(fold+1)*(len(train_y)/n_folds))
	train_idx=list(set(all_list)-set(valid_idx))
	cv_train_x=train_x[train_idx,:]
	cv_train_y=train_y[train_idx]
	cv_valid_x=train_x[valid_idx,:]
	cv_valid_y=train_y[valid_idx]
	knn.fit(cv_train_x,cv_train_y)
	pred_valid=knn.predict(cv_valid_x)
	valid_error[fold]=(sum(pred_valid!=cv_valid_y)+0.0)/len(pred_valid)

cv_error=sum(valid_error)/len(valid_error)
knn.fit(train_x,train_y)
pred_train=knn.predict(train_x)
train_error=(sum(pred_train!=train_y)+0.0)/len(pred_train)
pred_test=knn.predict(test_x)
test_error=(sum(pred_test!=test_y)+0.0)/len(pred_test)

x1_min, x1_max = train_x[:,0].min() - 0.05, train_x[:,0].max() + 0.05
x2_min, x2_max = train_x[:,1].min() - 0.05, train_x[:,1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x1_min, x1_max,0.01),np.arange(x2_min,x2_max,0.01))
Z_1 = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z_1 = Z_1.reshape(xx.shape)
plt.figure(1)
plt.pcolormesh(xx, yy, Z_1, cmap=cmap_light)
plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("KNN classification Result")
plt.xlabel("Symestry")
plt.ylabel("Density")
plt.show()
print "The Training Error is: ", train_error
print "The CV Error is: ", cv_error
print "The Testing Error is: ", test_error