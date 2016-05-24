import numpy as np
import matplotlib.pyplot as plt
import random

def data_gen(n,dim):
	data=np.zeros(shape=(n,dim))
	data[:,1:3]=np.random.rand(n,dim-1)
	return data

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

def substract_lists(a, b):
    for i, val in enumerate(a):
		val = val - b[i]
    return a

w_true=[3.0,-4.0,5.0]
'''
data=data_gen(20,3)
ind_true=np.sign(np.dot(data,w_true))

x_line=np.arange(0.0, 1.01, 0.01)
A=[x*(-w_true[1]/w_true[2]) for x in x_line]
B=len(x_line)*[w_true[0]/w_true[2]]
f_x=substract_lists(A, B)
plt.plot(x_line,f_x,'c',data[ind_true==1,1],data[ind_true==1,2],'r*',
	data[ind_true==-1,1],data[ind_true==-1,2],'bo')
plt.title("Original Data Points")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

iterations,w_b=pla(data,ind_true,[1.0,1.0,1.0])
print iterations

A_pla=[x*(-w_b[1]/w_b[2]) for x in x_line]
B_pla=len(x_line)*[w_b[0]/w_b[2]]
f_x_pla=substract_lists(A_pla, B_pla)
plt.plot(x_line,f_x,'c',x_line,f_x_pla,'g',data[ind_true==1,1],
	data[ind_true==1,2],'r*',data[ind_true==-1,1],data[ind_true==-1,2],'bo')
plt.title("Classification for Question b")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
'''
for size in [20,100,1000]:
	data=data_gen(size,3)
	ind_true=np.sign(np.dot(data,w_true))
	print ind_true==-1
	x_line=np.arange(0.0, 1.01, 0.01)
	A=[x*(-w_true[1]/w_true[2]) for x in x_line]
	B=len(x_line)*[w_true[0]/w_true[2]]
	f_x=substract_lists(A, B)
	iterations,w_b=pla(data,ind_true,[1.0,1.0,1.0])
	print "The number of iterations until convergence with size ",size," is:",iterations
	A_pla=[x*(-w_b[1]/w_b[2]) for x in x_line]
	B_pla=len(x_line)*[w_b[0]/w_b[2]]
	f_x_pla=substract_lists(A_pla, B_pla)
	plt.plot(x_line,f_x,'c',x_line,f_x_pla,'g',data[ind_true==1,1],
		data[ind_true==1,2],'r*',data[ind_true==-1,1],data[ind_true==-1,2],'bo')
	plt.title("Classification Results")
	plt.xlabel("X1")
	plt.ylabel("X2")
	plt.show()