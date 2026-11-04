import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('layoffs.csv')
df=df[['total_laid_off','funds_raised']].copy()
print("="*60)
print("1. DATASET OVERVIEW & STATISTICAL METRICS (BEFORE)")
print("="*60)
print(df.describe())
mean_val=df['total_laid_off'].mean()
std_val=df['total_laid_off'].std()
z_scores=(df['total_laid_off']-mean_val)/std_val
z_outliers=df[np.abs(z_scores)>2.5]
print(f"\nZ-Score Outliers (|Z| > 2.5): {len(z_outliers)}")
q1=df['total_laid_off'].quantile(0.25)
q3=df['total_laid_off'].quantile(0.75)
iqr=q3-q1
lower_bound=q1-1.5*iqr
upper_bound=q3+1.5*iqr
iqr_outliers=df[(df['total_laid_off']<lower_bound)|(df['total_laid_off']>upper_bound)]
print(f"IQR Outliers (Tukey's Fences): {len(iqr_outliers)}")
print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
df_removed=df[df['total_laid_off'].isna()|((df['total_laid_off']>=lower_bound)&(df['total_laid_off']<=upper_bound))].copy()
df_capped=df.copy()
df_capped['total_laid_off']=df_capped['total_laid_off'].clip(lower_bound,upper_bound)
df_transformed=df.copy()
df_transformed['total_laid_off']=np.log1p(df_transformed['total_laid_off'])
print("\n"+"="*60)
print("2. COMPARISON ACROSS TREATMENT TECHNIQUES")
print("="*60)
def calc_stats(s):
    return [len(s),s.mean(),s.std(),s.median(),s.quantile(0.75)-s.quantile(0.25)]
metrics=['Rows','Mean','Std','Median','IQR']
comp=pd.DataFrame({
    'Original':calc_stats(df['total_laid_off']),
    'Removed':calc_stats(df_removed['total_laid_off']),
    'Capped':calc_stats(df_capped['total_laid_off']),
    'Log-Transformed':calc_stats(df_transformed['total_laid_off'])
},index=pd.Index(metrics,name='Metric')).round(3)
print(comp)
fig,axes=plt.subplots(2,2,figsize=(12,8))
sns.boxplot(x=df['total_laid_off'],color='salmon',ax=axes[0,0])
axes[0,0].set_title("Box Plot: Before Treatment (Raw Outliers)")
sns.scatterplot(x='total_laid_off',y='funds_raised',data=df,hue=(np.abs(z_scores)>2.5),palette={True:'red',False:'blue'},ax=axes[0,1])
axes[0,1].set_title("Scatter Plot: Outliers (Z-Score)")
sns.boxplot(x=df_capped['total_laid_off'],color='lightgreen',ax=axes[1,0])
axes[1,0].set_title("Box Plot: After Capping (Winsorization)")
sns.histplot(df_transformed['total_laid_off'],kde=True,color='purple',ax=axes[1,1])
axes[1,1].set_title("Histogram + KDE: Log Transformed")
plt.tight_layout()
plt.show()
