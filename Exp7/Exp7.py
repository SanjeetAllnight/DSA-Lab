
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(os.path.join(os.path.dirname(__file__),'insurance.csv'))
print("="*60)
print("1. DATASET OVERVIEW")
print("="*60)
print("Dataset Shape:",df.shape)
print("\nFirst 5 Records:")
print(df.head())
print("\nMissing Values:",df.isnull().sum().sum())
print("\nStatistical Summary:")
print(df.describe())
num_features=['age','bmi','children']
cat_features=['sex','smoker','region']
features=num_features+cat_features
target='charges'
df=df.dropna(subset=features+[target])
X=df[features]
y=df[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("\nTraining Records:",len(X_train))
print("Testing Records:",len(X_test))
def preprocess(scale=True):
    return ColumnTransformer([
        ('num',StandardScaler() if scale else 'passthrough',num_features),
        ('cat',OneHotEncoder(handle_unknown='ignore'),cat_features)
    ])
models={
    'Simple Linear Regression':(Pipeline([('scale',StandardScaler()),('model',LinearRegression())]),['age']),
    'Multiple Linear Regression':(Pipeline([('preprocess',preprocess()),('model',LinearRegression())]),features),
    'Polynomial Regression':(Pipeline([('poly',PolynomialFeatures(degree=2,include_bias=False)),('scale',StandardScaler()),('model',LinearRegression())]),['age']),
    'Decision Tree Regression':(Pipeline([('preprocess',preprocess(False)),('model',DecisionTreeRegressor(random_state=42))]),features),
    'Random Forest Regression':(Pipeline([('preprocess',preprocess(False)),('model',RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1))]),features)
}
print("\n"+"="*60)
print("2. REGRESSION MODEL EVALUATION")
print("="*60)
results=[]
predictions={}
for name,(model,cols) in models.items():
    model.fit(X_train[cols],y_train)
    pred=model.predict(X_test[cols])
    predictions[name]=pred
    mse=mean_squared_error(y_test,pred)
    results.append([name,mean_absolute_error(y_test,pred),mse,np.sqrt(mse)])
comparison=pd.DataFrame(results,columns=['Model','MAE','MSE','RMSE']).set_index('Model').round(3)
print(comparison.sort_values('RMSE'))
best=comparison['RMSE'].idxmin()
print("\nBest Performing Model:",best)
print("\n"+"="*60)
print("3. ACTUAL VS PREDICTED VISUALIZATION")
print("="*60)
fig,axes=plt.subplots(2,3,figsize=(16,10))
for ax,(name,pred) in zip(axes.flat, predictions.items()):
    ax.scatter(y_test,pred,alpha=0.6)
    low=min(y_test.min(),pred.min())
    high=max(y_test.max(),pred.max())
    ax.plot([low,high],[low,high],'r--')
    ax.set_title(name)
    ax.set_xlabel("Actual Charges")
    ax.set_ylabel("Predicted Charges")
axes.flat[5].axis('off')
plt.tight_layout()
plt.show()