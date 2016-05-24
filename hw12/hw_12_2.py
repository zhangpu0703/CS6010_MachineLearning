import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
#from sknn.mlp import Classifier, Layer

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
#Question A 
train_error=[]
Iter=np.linspace(10,1e4,25)
for it in Iter:
	it=int(it)
	print it
	clf =MLPClassifier(algorithm='adam', alpha=0,hidden_layer_sizes=(10,),
	 max_iter=it, early_stopping=False)
	clf.fit(train_x,train_y)
	#clf=nn.BernoulliRBM(n_components=2)
	#clf=Classifier(layers=[Layer("Sigmoid", units=10),Layer("Linear")],
		#n_iter=20,regularize="L2",weight_decay=0.0)

	#clf.fit(train_x,train_y)

	yy=clf.predict(train_x)
	train_error.append((sum(yy!=train_y)+0.0)/len(yy))

plt.figure()
plt.plot(Iter,train_error,'r-')
plt.title("Training Error vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Training Classification Error")
plt.show()


#Question B 
clf =MLPClassifier(algorithm='adam',alpha=0,hidden_layer_sizes=(100,),
	early_stopping=False,validation_fraction=0.0,max_iter=300)
clf.fit(train_x,train_y)

x1_min, x1_max = train_x[:,0].min() - 0.05, train_x[:,0].max() + 0.05
x2_min, x2_max = train_x[:,1].min() - 0.05, train_x[:,1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x1_min, x1_max,0.01),np.arange(x2_min,x2_max,0.01))
Z_1 = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_1 = Z_1.reshape(xx.shape)

print clf.score(train_x,train_y)
plt.figure(1)
plt.pcolormesh(xx, yy, Z_1, cmap=cmap_light)
plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("MLP classification Result")
plt.xlabel("Symestry")
plt.ylabel("Density")
plt.show()

clf_reg = MLPClassifier(algorithm='adam',alpha=0,hidden_layer_sizes=(20,),
	early_stopping=False,validation_fraction=0.0,max_iter=300)
clf_reg.fit(train_x,train_y)
Z_1 = clf_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z_1 = Z_1.reshape(xx.shape)

print clf_reg.score(train_x,train_y)
plt.figure(2)
plt.pcolormesh(xx, yy, Z_1, cmap=cmap_light)
plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("MLP classification Result with Regularization")
plt.xlabel("Symestry")
plt.ylabel("Density")
plt.show()

'''
#Question C 
clf =MLPClassifier(algorithm='l-bfgs',alpha=0,hidden_layer_sizes=(10,),
	early_stopping=True,validation_fraction=0.1,max_iter=500)
clf.fit(train_x,train_y)
x1_min, x1_max = train_x[:,0].min() - 0.05, train_x[:,0].max() + 0.05
x2_min, x2_max = train_x[:,1].min() - 0.05, train_x[:,1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x1_min, x1_max,0.01),np.arange(x2_min,x2_max,0.01))
Z_1 = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_1 = Z_1.reshape(xx.shape)

print "The testing error is: ", 1-clf.score(test_x,test_y)
plt.figure(1)
plt.pcolormesh(xx, yy, Z_1, cmap=cmap_light)
plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("MLP classification Result with Early Stopping and Validation")
plt.xlabel("Symestry")
plt.ylabel("Density")
plt.show()
