
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
df=pd.read_csv(os.path.join(os.path.dirname(__file__),'Mall_Customers.csv'))
features=['Annual Income (k$)','Spending Score (1-100)']
data=df[features].dropna()
print("="*60)
print("1. DATASET OVERVIEW")
print("="*60)
print("Dataset Shape:",df.shape)
print("\nFirst 5 Records:")
print(df.head())
print("\nSelected Features:",features)
print("Missing Values:",data.isnull().sum().sum())
scaler=StandardScaler()
X_scaled=scaler.fit_transform(data)
print("\n"+"="*60)
print("2. ELBOW METHOD")
print("="*60)
wcss=[]
k_range=range(2,9)
for k in k_range:
    model=KMeans(n_clusters=k,random_state=42,n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)
plt.figure(figsize=(7,4))
plt.plot(k_range,wcss,'bo-',linewidth=2,markersize=7)
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares")
plt.grid(True,linestyle='--',alpha=0.6)
plt.tight_layout()
plt.show()
kmeans=KMeans(n_clusters=5,random_state=42,n_init=10)
kmeans_labels=kmeans.fit_predict(X_scaled)
hierarchical=AgglomerativeClustering(n_clusters=5,linkage='ward')
hier_labels=hierarchical.fit_predict(X_scaled)
dbscan=DBSCAN(eps=0.35,min_samples=5)
dbscan_labels=dbscan.fit_predict(X_scaled)
print("\n"+"="*60)
print("3. CLUSTER VISUALIZATION")
print("="*60)
fig,axes=plt.subplots(1,3,figsize=(18,5))
axes[0].scatter(X_scaled[:,0],X_scaled[:,1],c=kmeans_labels,cmap='viridis',edgecolor='k',s=50)
axes[0].scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='red',marker='X',label='Centroids')
axes[0].set_title("K-Means Clustering (K=5)")
axes[0].set_xlabel(features[0])
axes[0].set_ylabel(features[1])
axes[0].legend()
axes[1].scatter(X_scaled[:,0],X_scaled[:,1],c=hier_labels,cmap='plasma',edgecolor='k',s=50)
axes[1].set_title("Hierarchical Clustering (K=5)")
axes[1].set_xlabel(features[0])
axes[1].set_ylabel(features[1])
axes[2].scatter(X_scaled[:,0],X_scaled[:,1],c=dbscan_labels,cmap='coolwarm',edgecolor='k',s=50)
axes[2].set_title("DBSCAN Clustering")
axes[2].set_xlabel(features[0])
axes[2].set_ylabel(features[1])
plt.tight_layout()
plt.show()
print("\n"+"="*60)
print("4. CLUSTER EVALUATION METRICS")
print("="*60)
algorithms={'K-Means (K=5)':kmeans_labels,'Hierarchical (K=5)':hier_labels,'DBSCAN':dbscan_labels}
eval_results=[]
for name,labels in algorithms.items():
    mask=labels!=-1
    valid_labels=labels[mask]
    valid_data=X_scaled[mask]
    n_clusters=len(set(valid_labels))
    noise=np.sum(labels==-1)
    if n_clusters>1 and len(valid_data)>n_clusters:
        sil=silhouette_score(valid_data,valid_labels)
        db=davies_bouldin_score(valid_data,valid_labels)
        ch=calinski_harabasz_score(valid_data,valid_labels)
    else:
        sil,db,ch=np.nan,np.nan,np.nan
    eval_results.append([name,n_clusters,noise,sil,db,ch])
results_df=pd.DataFrame(eval_results,columns=['Algorithm','Clusters','Noise Points','Silhouette Score','Davies-Bouldin Index','Calinski-Harabasz Index'])
print(results_df.round(3).to_string(index=False))
