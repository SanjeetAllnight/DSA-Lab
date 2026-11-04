
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df=pd.read_csv(url,sep=';')
print("="*60)
print("1. DATASET OVERVIEW")
print("="*60)
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Description:")
print(df.describe())
print("\nNumerical Attributes:")
print(df.select_dtypes(include='number').columns)
print("\nMissing Values:")
print(df.isnull().sum().sum())
covariance=df.cov(numeric_only=True)
print("\n"+"="*60)
print("2. COVARIANCE MATRIX")
print("="*60)
print(covariance)
correlation=df.corr(numeric_only=True)
print("\n"+"="*60)
print("3. CORRELATION MATRIX")
print("="*60)
print(correlation)
plt.figure(figsize=(12,8))
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt='.2f')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
print("\nStrong Positive Correlations:")
found=False
for i in range(len(correlation.columns)):
    for j in range(i+1,len(correlation.columns)):
        if correlation.iloc[i,j]>0.5:
            print(correlation.columns[i],"and",correlation.columns[j],round(correlation.iloc[i,j],3))
            found=True
if not found:
    print("None")
print("\nStrong Negative Correlations:")
found=False
for i in range(len(correlation.columns)):
    for j in range(i+1,len(correlation.columns)):
        if correlation.iloc[i,j]<-0.5:
            print(correlation.columns[i],"and",correlation.columns[j],round(correlation.iloc[i,j],3))
            found=True
if not found:
    print("None")
print("\nHighly Correlated Attributes (Multicollinearity Check):")
features=correlation.drop(columns=['quality']).drop(index=['quality'])
found=False
for i in range(len(features.columns)):
    for j in range(i+1,len(features.columns)):
        if abs(features.iloc[i,j])>0.8:
            print(features.columns[i],"and",features.columns[j],round(features.iloc[i,j],3))
            found=True
if not found:
    print("No highly correlated pairs found")
print("\nCorrelation with Quality:")
print(correlation["quality"].sort_values(ascending=False))

