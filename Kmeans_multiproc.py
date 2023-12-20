## Import packages
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import seaborn as sns



## Load the .csv file
df=pd.read_csv('computerstocho.csv', delimiter = ',')
new_df=df.iloc[:,1:]
names = new_df.columns.tolist()
new_df.replace({'cd': {'no': 0, 'yes': 1}, 'laptop': {'no': 0, 'yes': 1}}, inplace=True)
data=np.array(new_df)


## Cluster computing function
def kmeans(k,X,max_iterations=200):
    it = 1

    ## Fix the seed to guarantee identical results
    np.random.seed(42) 
    
    ## Randomly choose rows of the data as first guess centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False), :]

    for _ in range(max_iterations):
        cluster_indices = []
        cluster_centers = []

        ## Divide the data into clusters based on the distance
        ## from the data to the centroids
        distances = cdist(X,centroids)
        belong_to_cluster = np.argmin(distances,axis=1)
        dist_min = distances[np.arange(len(distances)), belong_to_cluster]
        
        ## Create a list of arrays that shows the indices of the data
        ## points belonging to each cluster
        for i in range(k):
            cluster_indices.append(np.argwhere(belong_to_cluster==i))

        ## Compute the cluster center
        for indices in cluster_indices:
            cluster_centers.append(np.mean(X[indices], axis=0)[0])
        cluster_centers = np.array(cluster_centers)

        ## Relocate the centroids as the current cluster center
        ## or stop the code due to achieved convergence
        if np.max(centroids-cluster_centers)<0.001:
            break
        else:
            centroids = cluster_centers
            it += 1

    ## Return the division of data into clusters
    return belong_to_cluster, cluster_indices, centroids, np.sum(dist_min**2), it
    
## Initialize values for optimal k calculation
K = np.arange(1,13)
WCSS=np.zeros(np.max(K))

## Set function for average price calculation
def price_avg(X,c_i,k):
        prices = []
        for data_point in c_i[k]:
            prices.append(X[data_point,0])
        return np.mean(prices)


## Start timer
start_time = time.time()

if __name__ == "__main__":  
    
    pool = mp.Pool(4)

    ## Run kmeans for k clusters 
    test_for_k = pool.starmap(kmeans,[(k,data) for k in K])
    for k in (K-1):
        WCSS[k] = test_for_k[k][3] 
    
    pool.close()

    ## Compute the optimal k using the minimum of WCSS 3rd derivative
    first_derivative = np.gradient(WCSS)
    second_derivative = np.gradient(first_derivative)
    third_derivative = np.gradient(second_derivative)
    optimal_k = np.argmin(third_derivative)+int(len(K)/6)
    print("Optimal number of clusters:", optimal_k+1)

    ## Get the cluster division for the optimal k
    results = test_for_k[optimal_k]
    print('Convergence for ', optimal_k+1, ' clusters achieved in ', results[4], ' iterations')

    ## Find the cluster with the highest average price
    price_avgs = [price_avg(data,results[1],k) for k in range(optimal_k+1)]
    print('The cluster with the highest average price is cluster number ', np.argmax(price_avgs)+1)
    
    ## Stop timer
    end_time = time.time()

    ## Compute and print elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

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
    plt.scatter(data[:,0], data[:,1],c=labels)
    plt.scatter(centroids[:,0],centroids[:,1], marker='X',s=200,c='red')
    plt.show()

    ## Heatmap of the centroids and its coordinates
    norm_centroids = (centroids-centroids.min(axis=0))/(centroids.max(axis=0)-centroids.min(axis=0))
    x_axis_labels = names
    y_axis_labels = range(1,optimal_k+2)
    plot = sns.heatmap(norm_centroids, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    plot.set(title='Cluster coordinates')
    plt.show()