import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler,RobustScaler
df=pd.read_csv(os.path.join(os.path.dirname(__file__),'layoffs.csv'))
data=df[['total_laid_off','funds_raised']].dropna().copy()
print("="*60)
print("1. DATASET OVERVIEW")
print("="*60)
print("Original Records:",len(df))
print("Selected Features:",list(data.columns))
print("Records Used:",len(data))
print("\nOriginal Statistical Summary")
print(data.describe())
scaled_data={
    'Raw':data.copy(),
    'Min-Max':pd.DataFrame(MinMaxScaler().fit_transform(data),columns=data.columns,index=data.index),
    'Max-Abs':pd.DataFrame(MaxAbsScaler().fit_transform(data),columns=data.columns,index=data.index),
    'Z-Score':pd.DataFrame(StandardScaler().fit_transform(data),columns=data.columns,index=data.index),
    'Robust':pd.DataFrame(RobustScaler().fit_transform(data),columns=data.columns,index=data.index),
    'Mean-Norm':(data-data.mean())/(data.max()-data.min())
}
rows=[]
for feature in data.columns:
    for method,scaled in scaled_data.items():
        s=scaled[feature]
        rows.append([feature,method,s.mean(),s.std(),s.min(),s.max(),s.median(),s.quantile(.75)-s.quantile(.25)])
stats=pd.DataFrame(rows,columns=['Feature','Technique','Mean','Std','Min','Max','Median','IQR']).set_index(['Feature','Technique'])
print("\n"+"="*60)
print("2. STATISTICAL COMPARISON ACROSS SCALING TECHNIQUES")
print("="*60)
print(stats.round(3))
titles=['Raw Distribution','Min-Max Scaled [0,1]','Max-Abs Scaled','Z-Score Standardized','Robust Standardized (IQR)','Mean Normalized']
colors=['black','blue','cyan','green','orange','red']
for feature in data.columns:
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,(method,scaled),title,color in zip(axes.flat,scaled_data.items(),titles,colors):
        sns.kdeplot(x=scaled[feature],fill=True,color=color,ax=ax)
        ax.set_title(title)
    fig.suptitle(feature)
    plt.tight_layout()
    plt.show()