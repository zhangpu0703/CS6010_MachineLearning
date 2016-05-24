import numpy as np
import matplotlib.pyplot as plt
from random import *
from math import *

def two_coin(n,p,rep):
	freq=rep*[0.0]
	for i in range(rep):
		heads=0.0
		for j in range(n):
			mu=random()
			if mu>p:
				heads+=1.0
		freq[i]=heads/n


	return freq

N=int(1e5)
n=6
rep=2
p=0.5
first=[0]*N
second=[0]*N
for i in range(N):
	freq=two_coin(n,p,rep)
	first[i]=freq[0]
	second[i]=freq[1]


error=np.arange(0,1.01,0.01)
bounds=[0]*len(error)
p_true=[0]*len(error)

for i in range(len(error)):
	ep=error[i]
	l1=len([x for x in first if abs(x-p)>ep])
	l2=len([x for x in second if abs(x-p)>ep])
	p_true[i]=float(max(l1,l2))/N
	bounds[i]=4*exp(-2*(ep**2)*n)


plt.figure(1)
plt.plot(error,bounds,'r',error,p_true,'b-')
plt.ylabel('probability')
plt.legend(["Hoeffding Bound","True Probability"])
plt.show()
