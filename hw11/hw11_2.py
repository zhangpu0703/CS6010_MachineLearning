import numpy as np
import matplotlib.pyplot as plt
from math import *
from sklearn.cluster import KMeans
from numpy.linalg import inv
from matplotlib.colors import ListedColormap
import csv
from time import clock as now

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

def rbf_network(k,data,target):
	kmeans=KMeans(n_clusters=k)
	kmeans.fit(data)
	centers=kmeans.cluster_centers_
	Z=np.zeros((len(target),k))
	z=kmeans.predict(data)
	for i in range(len(target)):
		for j in range(k):
			d=sqrt((data[i,0]-centers[j,0])**2+
				(data[i,1]-centers[j,1])**2)
			Z[i,j]=1.0/(2*pi)*exp(-1.0/2*(d**2))

	Z=np.concatenate((np.ones((len(target),1)),Z), axis=1)
	w=np.dot(np.dot(inv(np.dot(Z.T,Z)),Z.T),target)

	return w,centers

def rbf_predict(w,data,centers):
	k=len(w)-1
	Z=np.zeros((len(data[:,0]),k))
	for i in range(len(data[:,0])):
		for j in range(k):
			d=sqrt((data[i,0]-centers[j,0])**2+
				(data[i,1]-centers[j,1])**2)
			Z[i,j]=1.0/(2*pi)*exp(-1.0/2*(d**2))

	Z=np.concatenate((np.ones((len(data[:,0]),1)),Z), axis=1)
	pred=np.sign(np.dot(Z,w))

	return pred

#error,pred=rbf_network(10,train_x,train_y)
#print len(pred), sum(pred==1)
'''
# Question A
size_K=30
cv_error=[0]*size_K
for k in range(1,size_K+1):
	all_list=range(len(train_x))
	n_folds=3
	valid_error=[0]*n_folds
	for fold in range(n_folds):
		valid_idx=range(fold*(len(train_x)/n_folds),(fold+1)*(len(train_x)/n_folds))
		train_idx=list(set(all_list)-set(valid_idx))
		cv_train_x=train_x[train_idx]
		cv_train_y=train_y[train_idx]
		cv_valid_x=train_x[valid_idx]
		cv_valid_y=train_y[valid_idx]
		start=now()
		w,centers=rbf_network(k,cv_train_x,cv_train_y)
		print "Train time:", now()-start
		start=now()
		pred_valid=rbf_predict(w,cv_valid_x,centers)
		print "Test time:", now()-start
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
k=14
all_list=range(len(train_x))
n_folds=3
valid_error=[0]*n_folds
for fold in range(n_folds):
	valid_idx=range(fold*(len(train_x)/n_folds),(fold+1)*(len(train_x)/n_folds))
	train_idx=list(set(all_list)-set(valid_idx))
	cv_train_x=train_x[train_idx]
	cv_train_y=train_y[train_idx]
	cv_valid_x=train_x[valid_idx]
	cv_valid_y=train_y[valid_idx]
	w,centers=rbf_network(k,cv_train_x,cv_train_y)
	pred_valid=rbf_predict(w,cv_valid_x,centers)
	valid_error[fold]=(sum(pred_valid!=cv_valid_y)+0.0)/len(pred_valid)

cv_error=sum(valid_error)/len(valid_error)

w,centers=rbf_network(k,train_x,train_y)
pred_train=rbf_predict(w,train_x,centers)
train_error=(sum(pred_train!=train_y)+0.0)/len(train_y)
pred_test=rbf_predict(w,test_x,centers)
test_error=(sum(pred_test!=test_y)+0.0)/len(test_y)
x1_min, x1_max = train_x[:,0].min() - 0.05, train_x[:,0].max() + 0.05
x2_min, x2_max = train_x[:,1].min() - 0.05, train_x[:,1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x1_min, x1_max,0.01),np.arange(x2_min,x2_max,0.01))
Z_1=pred_train=rbf_predict(w,np.c_[xx.ravel(), yy.ravel()],centers)
Z_1 = Z_1.reshape(xx.shape)
plt.figure(1)
plt.pcolormesh(xx, yy, Z_1, cmap=cmap_light)
plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("RBF Network classification Result")
plt.xlabel("Symestry")
plt.ylabel("Density")
plt.show()
print "The Training Error is: ", train_error
print "The CV Error is: ", cv_error
print "The Testing Error is: ", test_error