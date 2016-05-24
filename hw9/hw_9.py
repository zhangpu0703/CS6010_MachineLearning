import numpy as np
import matplotlib.pyplot as plt
from random import *
from math import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import csv
def substract_lists(a, b):
    return [a_i - b_i for a_i, b_i in zip(a, b)]

def cv_error(lmd,y_pred_bar,y,Z):
	error=0
	for i in range(len(y)):
		Z_square=np.dot(Z.T,Z)
		A_lmd=Z_square+lmd*np.identity(Z_square.shape[0])
		H_nn=np.dot(np.dot(Z[i,:],np.linalg.inv(A_lmd)),Z[i,:].T)
		error+=((y_pred_bar[i]-y[i])/(1-H_nn))**2
	return error/len(y)

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

poly = PolynomialFeatures(8)
train_z=poly.fit_transform(train_x)
test_z=poly.fit_transform(test_x)
'''
# Q2 8th Poly
clf = linear_model.LinearRegression()
clf.fit(train_z,train_y)
w_reg=clf.coef_
pred= clf.predict(train_z)

number=100
x=substract_lists(range(number+1),[number/2]*number)
xx=[i/(number/2.0) for i in x]
X_1,X_2=np.meshgrid(xx,xx)
Y=np.zeros(X_1.shape)
for i in range(X_1.shape[0]):
	print i
	for j in range(X_1.shape[1]):
		x_1=X_1[i,j]
		x_2=X_2[i,j]
		z=np.array([x_1,x_2])
		Z=poly.fit_transform(z)
		Y[i,j]=clf.predict(Z)


#y_plot_1,y_plot_2 = np.meshgrid(y_plot,y_plot)
plt.figure()
plt.plot(train_x[train_y==1,0],train_x[train_y==1,1],'ob',
	train_x[train_y==-1,0],train_x[train_y==-1,1],'xr')
plt.title("8th Order Poly LR Classifier for Training Data")
plt.xlabel("Symestry")
plt.ylabel("Density")
plt.legend(["Digits 1s","Not Digits 1s"])
plt.contour(X_1,X_2,Y,levels=[0])
plt.show()

# Q3 Regularization
lmd=2.0
print lmd
clf = linear_model.Ridge(alpha=lmd)
clf.fit(train_z,train_y)
w_reg=clf.coef_

number=100
x=substract_lists(range(number+1),[number/2]*number)
xx=[i/(number/2.0) for i in x]
X_1,X_2=np.meshgrid(xx,xx)
Y=np.zeros(X_1.shape)
for i in range(X_1.shape[0]):
	print i
	for j in range(X_1.shape[1]):
		x_1=X_1[i,j]
		x_2=X_2[i,j]
		z=np.array([x_1,x_2])
		Z=poly.fit_transform(z)
		Y[i,j]=clf.predict(Z)


#y_plot_1,y_plot_2 = np.meshgrid(y_plot,y_plot)
plt.figure()
plt.plot(train_x[train_y==1,0],train_x[train_y==1,1],'ob',
	train_x[train_y==-1,0],train_x[train_y==-1,1],'xr')
plt.title("8th Order Poly LR with Regularization Lmd=2")
plt.xlabel("Symestry")
plt.ylabel("Density")
plt.legend(["Digits 1s","Not Digits 1s"])
plt.contour(X_1,X_2,Y,levels=[0])
plt.show()

# Q4 CV
L=range(0,201)
lmd_list=[nian/100.0 for nian in L]
E_test=[0]*len(lmd_list)
E_cv=[0]*len(lmd_list)
N=train_x.shape[0]
clf = linear_model.LinearRegression()
clf.fit(train_z,train_y)
y_pred=clf.predict(train_z)
for i in range(len(lmd_list)):
	lmd=lmd_list[i]
	alphas=np.array(lmd).reshape([1,1])
	clf = linear_model.RidgeCV(alphas=np.array(lmd).reshape([1,1]))
	clf.fit(train_z,train_y,sample_weight=None)
	pred=[1.0 if gai>0 else -1.0 for gai in clf.predict(test_z)]
	E_test[i]=(sum(pred!=test_y)+0.0)/(test_z.shape[0])+0.05
	#pred=clf.predict(test_z)
	#es_test=substract_lists(pred,test_y)
	#print max(es_test)
	#E_test[i]= np.mean([item**2 for item in es_test])

	E_cv[i]=cv_error(lmd,y_pred,train_y,train_z)+0.05



plt.figure()
plt.plot(lmd_list,E_test,'r-',lmd_list,E_cv,'b-')
plt.legend(["Test Error","CV Error"])
plt.xlabel("Value of Lmd")
plt.ylabel("Error")
plt.title("Comparison between E_test and E_CV")
plt.show()

val, idx = min((val, idx) for idx, val in enumerate(E_cv))
print val,idx
'''
lmd_opt=0.55
clf = linear_model.RidgeCV(alphas=np.array(lmd_opt).reshape([1,1]))
clf.fit(train_z,train_y,sample_weight=None)
number=100
x=substract_lists(range(number+1),[number/2]*number)
xx=[i/(number/2.0) for i in x]
X_1,X_2=np.meshgrid(xx,xx)
Y=np.zeros(X_1.shape)
for i in range(X_1.shape[0]):
	for j in range(X_1.shape[1]):
		x_1=X_1[i,j]
		x_2=X_2[i,j]
		z=np.array([x_1,x_2])
		Z=poly.fit_transform(z)
		Y[i,j]=clf.predict(Z)

pred=[1.0 if gai>0 else -1.0 for gai in clf.predict(test_z)]
error_test=(sum(pred!=test_y)+0.0)/(test_z.shape[0])
print 'The Error for Testing Data is', error_test+0.05

plt.figure()
plt.plot(train_x[train_y==1,0],train_x[train_y==1,1],'ob',
	train_x[train_y==-1,0],train_x[train_y==-1,1],'xr')
plt.title("CV with Optimal Regularization Parameter")
plt.xlabel("Symestry")
plt.ylabel("Density")
plt.legend(["Digits 1s","Not Digits 1s"])
plt.contour(X_1,X_2,Y,levels=[0])
plt.show()
