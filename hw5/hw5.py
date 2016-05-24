import numpy as np
import matplotlib.pyplot as plt
from random import *
from math import *

def substract_lists(a, b):
    return [a_i - b_i for a_i, b_i in zip(a, b)]

R=1000
gr_1=[0]*R
gr_2=[0]*R
error=[0]*R
var=[0]*R
for r in range(R):
	x_1,x_2=uniform(-1,1),uniform(-1,1)
	gr_1[r]=x_1+x_2
	gr_2[r]=x_1*x_2
	
g_bar=[np.mean(gr_1),np.mean(gr_2)]

num_sample=2000
sample=sorted([uniform(-1,1) for i in range(num_sample)])

g_bar_sam=[0]*num_sample
f_true=[0]*num_sample
for j in range(num_sample):
	g_bar_sam[j]=g_bar[0]*sample[j]-g_bar[1]
	f_true[j]=sample[j]**2

for i in range(R):
	g_est=[0]*num_sample
	for j in range(num_sample):
		g_est[j]=gr_1[i]*sample[j]-gr_2[i]

	erroria=substract_lists(g_est,f_true)
	error[i]=np.mean([item**2 for item in erroria])
	varia=substract_lists(g_est,g_bar_sam)
	var[i]=np.mean([item**2 for item in varia])

bias_1=substract_lists(g_bar_sam,f_true)
bias=np.mean([item**2 for item in bias_1])

print "The Bias is:", bias
print "The Variance is:", np.mean(var)
print "The E_out is:", np.mean(error)

plt.figure(1)
plt.plot(sample,f_true,'r',sample,g_bar_sam,'b')
plt.title("Comparison between f and g")
plt.legend(['True function','Estimated function'])
plt.xlabel('X')
plt.xlabel('Y')
plt.show()
