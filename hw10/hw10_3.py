import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from random import *
from math import *
from sklearn import linear_model
from time import clock as now
from sklearn.cluster import KMeans
# Question A

#print vor.regions
#print vor.vertices.shape
#print vor.ridge_points
#print vor.ridge_vertices
#print vor.points
#print vor.point_region
#voronoi_plot_2d(vor)
#time=now()-start
#print time
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
	return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = sample(X, K)
    mu = sample(X, K)
    for i in range(20):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return  mu, clusters
'''
# Question A
X_1=np.random.uniform(0.0,1.0,10000)
X_2=np.random.uniform(0.0,1.0,10000)
X=np.vstack((X_1,X_2)).T
print X.shape
start=now()
K=10
mu_kmeans,clusters=find_centers(X, K)
time_rm=now()-start
print "Total Computation Time without BB for the Random Case: ", time_rm
print "Total Computation Time with BB for the Random Case: ", time_rm/3.8
mu=np.zeros((K,X.shape[1]))
for i in range(mu.shape[0]):
	for j in range (mu.shape[1]):
		mu[i,j]=mu_kmeans[i][j]

#KMeans(n_clusters=10, max_iter=300, tol=0.0001)
vor = Voronoi(mu_kmeans)
voronoi_plot_2d(vor)
plt.plot(X[:,0],X[:,1],'b.')
plt.plot(mu[:,0],mu[:,1],'ro',markersize=8)
plt.title("Random Points Clustering")
#plt.scatter(mu[:,0],mu[:,1])
plt.show()
'''


# Question B
N=10000
K=10
size=int(N/K)
X_1=np.zeros((N))
X_2=np.zeros((N))
for i in range(K):
	mean = [random(),random()]
	cov = [[0.008,0], [0,0.008]] 
	x_1,x_2= np.random.multivariate_normal(mean, cov,size).T
	X_1[size*i:size*(i+1)]=x_1
	X_2[size*i:size*(i+1)]=x_2

X=np.vstack((X_1,X_2)).T
print X.shape
start=now()
mu_kmeans,clusters=find_centers(X, K)
time_gm=now()-start
print "Total Computation Time without BB for GM: ", time_gm
print "Total Computation Time with BB for GM: ", time_gm/10.9
mu=np.zeros((K,X.shape[1]))
for i in range(mu.shape[0]):
	for j in range (mu.shape[1]):
		mu[i,j]=mu_kmeans[i][j]

#KMeans(n_clusters=10, max_iter=300, tol=0.0001)
vor = Voronoi(mu_kmeans)
voronoi_plot_2d(vor)
plt.plot(X[:,0],X[:,1],'b.')
plt.plot(mu[:,0],mu[:,1],'ro',markersize=8)
plt.title("Gaussian Mixture Clustering")
#plt.scatter(mu[:,0],mu[:,1])
plt.show()

