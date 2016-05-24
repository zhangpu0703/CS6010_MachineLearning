import numpy as np
import matplotlib.pyplot as plt
from random import *
from math import *

def gd(num_iter,ita,initial):
	cur=initial
	F_iters=[cur[0]**2+2*cur[1]**2+2*sin(2*pi*cur[0])*sin(2*pi*cur[1])]
	for i in range(num_iter):
		grad_x=2*cur[0]+4*pi*(cos(2*pi*cur[0])*sin(2*pi*cur[1]))
		grad_y=4*cur[1]+4*pi*(sin(2*pi*cur[0])*cos(2*pi*cur[1]))
		cur[0]=cur[0]-ita*grad_x
		cur[1]=cur[1]-ita*grad_y
		f_val=cur[0]**2+2*cur[1]**2+2*sin(2*pi*cur[0])*sin(2*pi*cur[1])
		F_iters.append(f_val)

	return cur,F_iters,f_val

initial=[0.1,0.1]
ita=0.01
num_iter=50

for gainian in [0.1,1.0,-0.5,-1.0]:
	initial=[gainian,gainian]
	cur,F_iters,f_val=gd(num_iter,ita,initial)
	print cur, f_val

plt.plot(range(num_iter+1),F_iters,'ro-')
plt.title("Values of f When Ita = 0.01")
plt.xlabel("Iterations")
plt.ylabel("Values")
plt.show()

