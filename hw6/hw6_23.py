import numpy as np
import matplotlib.pyplot as plt
from random import *
from math import *
from sklearn import linear_model

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
'''
data,indx=data_gen(5,5,10,2000,0.5,0,0)
regr = linear_model.LinearRegression()
xx=data.reshape(2000,3)
yy=indx
regr.fit(xx,yy)
w_lr=regr.coef_
print 'LR Coefficients:', regr.coef_
'''

def substract_lists(a, b):
    for i, val in enumerate(a):
		val = val - b[i]
    return a

def pla(data,ind_true,w_ini):
	w_cur=w_ini
	y_cur=np.sign(np.dot(data,w_cur))
	iterations=0
	while not (y_cur == ind_true).all():
		iterations+=1
		misclass=np.where(y_cur != ind_true)
		imp=misclass[0][0]
		w_cur=w_cur+ind_true[imp]*data[imp,:]
		y_cur=np.sign(np.dot(data,w_cur))

	return iterations,w_cur

sep_dif=np.arange(0.2,5.1,0.2)
Iter=[0]*len(sep_dif)
for t in range(len(sep_dif)):
	sep=sep_dif[t]
	data,ind_true=data_gen(5,sep,10,2000,0.5,0,0)
	ind_true=np.asarray(ind_true)
	iterations,w_b=pla(data,ind_true,[0.001,0.001,0.001])
	Iter[t]=iterations

plt.plot(sep_dif,Iter,'ro-')
plt.title("Iterations v.s. Sep")
plt.xlabel("Sep")
plt.ylabel("Iterations")
plt.show()

'''
iterations,w_b=pla(data,ind_true,[0.001,0.001,0.001])
print iterations
print w_b
x_line=np.arange(0.0, 50.0, 0.1)
A_pla=[x*(-w_b[1]/w_b[2]) for x in x_line]
B_pla=len(x_line)*[w_b[0]/w_b[2]]
f_x_pla=substract_lists(A_pla, B_pla)
A_lr=[x*(-w_lr[1]/w_lr[2]) for x in x_line]
B_lr=len(x_line)*[w_lr[0]/w_lr[2]]
f_x_lr=substract_lists(A_lr, B_lr)
plt.plot(x_line,f_x_pla,'g',data[ind_true==-1,1],data[ind_true==-1,2],'r*',data[ind_true==1,1],data[ind_true==1,2],'bo')
plt.title("PLA Classification Results")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
plt.plot(x_line,f_x_lr,'g',data[ind_true==-1,1],data[ind_true==-1,2],'r*',data[ind_true==1,1],data[ind_true==1,2],'bo')
plt.title("Linear Regression Results")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
'''