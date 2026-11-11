
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv(os.path.join(os.path.dirname(__file__),'data_banknote_authentication.txt'),header=None,names=['variance','skewness','curtosis','entropy','class'])

print("="*60)
print("1. DATASET OVERVIEW")
print("="*60)
print("Dataset Shape:",df.shape)
print("\nFirst 5 Records:")
print(df.head())
print("\nMissing Values:",df.isnull().sum().sum())
print("\nClass Distribution:")
print(df['class'].value_counts().sort_index())

X=df.drop(columns=["class"])
y=df["class"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

models={
    "Logistic Regression":LogisticRegression(max_iter=200,random_state=42),
    "K-Nearest Neighbours":KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":DecisionTreeClassifier(max_depth=3,random_state=42),
    "Support Vector Machine":SVC(kernel="rbf",random_state=42),
    "Random Forest":RandomForestClassifier(n_estimators=100,max_depth=3,random_state=42)
}

results=[]
confusion_matrices={}

for name,model in models.items():
    if name in ["Logistic Regression","K-Nearest Neighbours","Support Vector Machine"]:
        model.fit(X_train_scaled,y_train)
        y_pred=model.predict(X_test_scaled)
    else:
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)

    acc=accuracy_score(y_test,y_pred)
    prec=precision_score(y_test,y_pred,average="weighted")
    rec=recall_score(y_test,y_pred,average="weighted")
    f1=f1_score(y_test,y_pred,average="weighted")
    cm=confusion_matrix(y_test,y_pred,labels=[0,1])

    confusion_matrices[name]=cm
    results.append({"Model":name,"Accuracy":acc,"Precision":prec,"Recall":rec,"F1-Score":f1})

results_df=pd.DataFrame(results).sort_values(by="Accuracy",ascending=False)

print("\n"+"="*60)
print("2. MODEL EVALUATION SUMMARY")
print("="*60)
print(results_df.to_string(index=False))

fig,axes=plt.subplots(1,5,figsize=(22,4))
class_names=["Genuine","Forged"]

for ax,(name,cm) in zip(axes,confusion_matrices.items()):
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",cbar=False,xticklabels=class_names,yticklabels=class_names,ax=ax)
    ax.set_title(name,fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()
