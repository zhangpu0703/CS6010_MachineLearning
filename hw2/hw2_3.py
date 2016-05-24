import numpy as np
import matplotlib.pyplot as plt
from random import *
from math import *

def flip_coin(n,p,rep):
	freq=rep*[0.0]
	for i in range(rep):
		heads=0.0
		for j in range(n):
			mu=random()
			if mu>p:
				heads+=1.0
		freq[i]=heads/n

	v_1=freq[1]
	v_rand=freq[randint(0,rep-1)]
	v_min=min(freq)

	return v_1,v_rand,v_min

N=int(1e5)
n=10
rep=1000
p=0.5
first_freq=[0]*N
random_freq=[0]*N
min_freq=[0]*N
for i in range(N):
	v_1,v_rand,v_min=flip_coin(n,p,rep)
	first_freq[i]=v_1
	random_freq[i]=v_rand
	min_freq[i]=v_min

error=np.arange(0,1.01,0.01)
bounds=[0]*len(error)
p_1=[0]*len(error)
p_rand=[0]*len(error)
p_min=[0]*len(error)
for i in range(len(error)):
	ep=error[i]
	p_1[i]=float(len([x for x in first_freq if abs(x-p)>ep]))/N
	p_rand[i]=float(len([x for x in random_freq if abs(x-p)>ep]))/N
	p_min[i]=float(len([x for x in min_freq if abs(x-p)>ep]))/N
	bounds[i]=2*exp(-2*(ep**2)*n)

plt.figure(1)
plt.subplot(131)
plt.hist(first_freq)
plt.ylabel('Frequency')
plt.title('First Coin Histogram')
plt.subplot(132)
plt.hist(random_freq)
plt.ylabel('Frequency')
plt.title('Random Coin Histogram')
plt.subplot(133)
plt.hist(min_freq)
plt.ylabel('Frequency')
plt.title('Cmin Coin Histogram')
plt.show()

plt.figure(2)
plt.plot(error,bounds,'r',error,p_1,'b-')
plt.ylabel('probability')
plt.legend(["Hoeffding Bound","First Coin"])
plt.show()
plt.figure(3)
plt.plot(error,bounds,'r',error,p_rand,'b-')
plt.ylabel('probability')
plt.legend(["Hoeffding Bound","Random Coin"])
plt.show()
plt.figure(4)
plt.plot(error,bounds,'r',error,p_min,'b-')
plt.ylabel('probability')
plt.legend(["Hoeffding Bound","Cmin Coin"])
plt.show()