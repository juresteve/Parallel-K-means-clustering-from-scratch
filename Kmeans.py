# import packages
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns



# read and edit the csv file
df=pd.read_csv('computers.csv', delimiter = ',')
new_df=df.iloc[:,1:]
names = new_df.columns.tolist()
new_df.replace({'cd': {'no': 0, 'yes': 1}, 'laptop': {'no': 0, 'yes': 1}}, inplace=True)
datos=np.array(new_df)


def kmeans(k,X,max_iterations=200):
    it = 1
    #fijar semilla para comparar velocidades (quitar al acabar el estudio)
    np.random.seed(42) 
    # randomly choose rows of the data as first guess centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False), :]
    # divide the data for some iterations or until convergence
    for _ in range(max_iterations):
        cluster_indices=[]
        cluster_centers =[]
        # divide the data into clusters based on the distance
        # from the data to the centroids
        distances=cdist(X,centroids)
        belong_to_cluster=np.argmin(distances,axis=1)
        dist_min= distances[np.arange(len(distances)), belong_to_cluster]
        # create a list of lists that shows the indeces of the rows
        # of data set into each cluster
        for i in range(k):
            cluster_indices.append(np.argwhere(belong_to_cluster==i))
        # find the cluster center
        for indices in cluster_indices:
            cluster_centers.append(np.mean(X[indices], axis=0)[0])
        cluster_centers=np.array(cluster_centers)
        # relocate the centroids as the current cluster center
        # or stop the code due to achieved convergence
        if np.max(centroids-cluster_centers)<0.001:
            break
        else:
            centroids=cluster_centers
            it += 1
    # return the division of data into clusters
    return belong_to_cluster, cluster_indices, centroids, np.sum(dist_min**2), it
## Set function for average price calculation
def price_avg(X,c_i,k):
        prices = []
        for data_point in c_i[k]:
            prices.append(X[data_point,0])
        return np.mean(prices)    
# start chronometer
resultados = {}
WCSS=[]
K = np.arange(1,13)
start_time = time.time()
for k in K:
# run the function for k clusters and fit the data
    resultados[f"resultados{k}"] =kmeans(k,datos)
    WCSS.append(resultados[f"resultados{k}"][3])
## Compute the optimal k using the minimum of WCSS 3rd derivative
first_derivative = np.gradient(WCSS)
second_derivative = np.gradient(first_derivative)
third_derivative = np.gradient(second_derivative)
optimal_k = np.argmin(third_derivative)+int(len(K)/6)
print("Optimal number of clusters:", optimal_k+1)

## Get the cluster division for the optimal k
results = resultados[f"resultados{optimal_k+1}"]
print('Convergence for ', optimal_k+1, ' clusters achieved in ', results[4], ' iterations')

## Find the cluster with the highest average price

price_avgs = [price_avg(datos,results[1],k) for k in range(optimal_k+1)]
print('The cluster with the highest average price is cluster number ', np.argmax(price_avgs)+1)
# stop chronometer
end_time = time.time()
# compute and print elapsed time
elapsed_time = end_time - start_time
print("Tiempo transcurrido:", elapsed_time, "segundos")

## Show the elbow graph
plt.scatter(K, WCSS, label='(k,WCSS) pairs', color='blue')
plt.plot(K, WCSS, label='Linear interpolation', color='red', linestyle='--')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow graph')
plt.legend()
plt.grid(True)
plt.show()

## Plot the first 2 dimensions of the data
labels = results[0]
centroids = results[2]
plt.scatter(datos[:,0], datos[:,1],c=labels,label='Data divided into clusters')
plt.scatter(centroids[:,0],centroids[:,1], marker='X',s=200,c='red',label='Centroids')
plt.xlabel('First dimension data')
plt.ylabel('Second dimension data')
plt.title('First two dimensions of data set')
plt.legend()
plt.grid(True)
plt.show()

## Heatmap of the centroids and its coordinates
norm_centroids = (centroids-centroids.min(axis=0))/(centroids.max(axis=0)-centroids.min(axis=0))
x_axis_labels = names
y_axis_labels = range(1,optimal_k+2)
plot = sns.heatmap(norm_centroids, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plot.set(title='Cluster coordinates')
plt.show()

