import numpy as np
import matplotlib.pyplot as plt

X_1,X_2=[1,0],[-1,0]
y_1,y_2=1,-1

w,b=[1,0],0

x=np.arange(-2.5,2.5,0.01)
xx,yy=np.meshgrid(x,x)
Y_ori=np.zeros(xx.shape)
Y_zspace=np.zeros(xx.shape)

for i in range(xx.shape[0]):
	#print i
	for j in range(xx.shape[1]):
		x_1=xx[i,j]
		x_2=yy[i,j]
		g_x=w[0]*x_1+w[1]*x_2+b
		Y_ori[i,j]=int(g_x>0)
		z_1=x_1**3-x_2
		z_2=x_1*x_2
		Y_zspace[i,j]=int(z_1>0)
		

plt.figure()
plt.plot(X_1[0],X_1[1],'ob', X_2[0],X_2[1],'xr')
plt.title("Decision Boundary in X Space and Z Space")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.contour(xx,yy,Y_ori,levels=[0],colors='r')
plt.contour(xx,yy,Y_zspace,levels=[0])
plt.show()