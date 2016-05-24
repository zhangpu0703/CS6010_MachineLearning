import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC 

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
# Question A and B
c_list=[0.01,10]
for c in c_list:
	c_at=c*300
	clf=SVC(kernel='poly', degree=8,C=c_at)
	clf.fit(train_x,train_y)

	x1_min, x1_max = train_x[:,0].min() - 0.05, train_x[:,0].max() + 0.05
	x2_min, x2_max = train_x[:,1].min() - 0.05, train_x[:,1].max() + 0.05
	xx, yy = np.meshgrid(np.arange(x1_min, x1_max,0.01),np.arange(x2_min,x2_max,0.01))
	Z_1 = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z_1 = Z_1.reshape(xx.shape)
	print clf.score(train_x,train_y)
	plt.figure()
	plt.pcolormesh(xx, yy, Z_1, cmap=cmap_light)
	plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=cmap_bold)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.title("SVM with 8th Order Polynomial Kernel")
	plt.xlabel("Symestry")
	plt.ylabel("Density")
	plt.show()


# Question C
c_list=np.arange(-2,2.01,0.1)

cv_error = [0]*len(c_list)
for i,c in enumerate(c_list):
	c_at=(10**c)*len(train_y)
	K=5
	train_ind=range(len(train_y))
	error_cv=0
	for j in range(K):
		clf=SVC(kernel='poly', degree=8,C=c_at)
		val_ind=range(j*len(train_y)/K+1,(j+1)*len(train_y)/K)
		fold_ind=list(set(train_ind) - set(val_ind))
		new_x=train_x[fold_ind,:]
		new_y=train_y[fold_ind]
		new_x_val=train_x[val_ind,:]
		new_y_val=train_y[val_ind]
		clf.fit(new_x,new_y)
		error_cv+=1-clf.score(new_x_val,new_y_val)

	cv_error[i]=error_cv/K

plt.figure()
plt.plot(c_list,cv_error,'ro-')
plt.title("CV Error vs C")
plt.xlabel("Value of log_10 of C")
plt.ylabel("CV Error")
plt.show()
'''

c=1.0
c_at=(10**c)*len(train_y)
clf=SVC(kernel='poly', degree=8,C=c_at)
clf.fit(train_x,train_y)
test_error=1-clf.score(test_x,test_y)

x1_min, x1_max = train_x[:,0].min() - 0.05, train_x[:,0].max() + 0.05
x2_min, x2_max = train_x[:,1].min() - 0.05, train_x[:,1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x1_min, x1_max,0.01),np.arange(x2_min,x2_max,0.01))
Z_1 = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_1 = Z_1.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, Z_1, cmap=cmap_light)
plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("SVM with 8th Order Polynomial Kernel at Optimal C")
plt.xlabel("Symestry")
plt.ylabel("Density")
plt.show()

print "The Testing Error is:", test_error







