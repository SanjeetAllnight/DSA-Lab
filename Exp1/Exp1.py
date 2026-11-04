import pandas as pd
import numpy as np
datasets={"Employee Attrition":"Employee_Attrition.csv","Retail Sales":"retail_sales_dataset.csv","Student":"student_data.csv"}
data={name:pd.read_csv(file) for name,file in datasets.items()}
for name,df in data.items():
    print("="*55)
    print(name)
    print("="*55)
    print("\n1. Records and Attributes")
    print("Records:",len(df),"Attributes:",len(df.columns))
    print("\n2. Data Types")
    types=df.dtypes.astype(str).value_counts()
    print("Numeric:",types.get("int64",0)+types.get("float64",0),"Categorical:",types.get("object",0))
    print("\n3. Statistical Summary")
    if name=="Employee Attrition":
        cols=["Age","MonthlyIncome","TotalWorkingYears","YearsAtCompany"]
    elif name=="Retail Sales":
        cols=["Age","Quantity","Price per Unit","Total Amount"]
    else:
        cols=["age","absences","G1","G2","G3"]
    print(df[cols].describe().round(2).to_string())
    print("\n4. Unique Values")
    unique=df.nunique()
    print("Minimum:",int(unique.min()),"Maximum:",int(unique.max()),"Average:",round(unique.mean(),2))
    print("\n5. Missing Values")
    print("Total Missing:",int(df.isnull().sum().sum()))
    print("\n6. Duplicate Records")
    print("Duplicates:",int(df.duplicated().sum()))
    print("\n7. Potential Outliers")
    found=False
    for col in df.select_dtypes(include=np.number):
        q1,q3=df[col].quantile([.25,.75])
        iqr=q3-q1
        n=int(((df[col]<q1-1.5*iqr)|(df[col]>q3+1.5*iqr)).sum())
        if n:
            print(col,":",n)
            found=True
    if not found:
        print("None")
    print("\n8. Incorrect Values")
    checks={}
    if name=="Employee Attrition":
        checks={
            "Age":((df["Age"]<18)|(df["Age"]>100)).sum(),
            "PercentSalaryHike":((df["PercentSalaryHike"]<0)|(df["PercentSalaryHike"]>100)).sum(),
            "EmployeeCount":(df["EmployeeCount"]!=1).sum(),
            "StandardHours":(df["StandardHours"]!=80).sum(),
            "YearsAtCompany > TotalWorkingYears":(df["YearsAtCompany"]>df["TotalWorkingYears"]).sum()
        }
    elif name=="Retail Sales":
        checks={
            "Age":((df["Age"]<0)|(df["Age"]>120)).sum(),
            "Quantity":(df["Quantity"]<=0).sum(),
            "Price per Unit":(df["Price per Unit"]<0).sum(),
            "Total Amount":(df["Total Amount"]!=df["Quantity"]*df["Price per Unit"]).sum(),
            "Invalid Date":pd.to_datetime(df["Date"],errors="coerce").isna().sum()
        }
    else:
        checks={
            "Age":((df["age"]<15)|(df["age"]>25)).sum(),
            "Medu":((df["Medu"]<0)|(df["Medu"]>4)).sum(),
            "Fedu":((df["Fedu"]<0)|(df["Fedu"]>4)).sum(),
            "G1":((df["G1"]<0)|(df["G1"]>20)).sum(),
            "G2":((df["G2"]<0)|(df["G2"]>20)).sum(),
            "G3":((df["G3"]<0)|(df["G3"]>20)).sum(),
            "Absences":(df["absences"]<0).sum()
        }
    found=False
    for col,n in checks.items():
        n=int(n)
        if n:
            print(col,":",n)
            found=True
    if not found:
        print("None")
    print("\n9. Inconsistent Values")
    checks={}
    if name=="Employee Attrition":
        checks={
            "Attrition":set(df["Attrition"].dropna().unique())-{"Yes","No"},
            "Gender":set(df["Gender"].dropna().unique())-{"Male","Female"},
            "OverTime":set(df["OverTime"].dropna().unique())-{"Yes","No"}
        }
    elif name=="Retail Sales":
        checks={
            "Gender":set(df["Gender"].dropna().unique())-{"Male","Female"},
            "Product Category":set(df["Product Category"].dropna().unique())-{"Beauty","Clothing","Electronics"}
        }
    else:
        checks={
            "school":set(df["school"].dropna().unique())-{"GP","MS"},
            "sex":set(df["sex"].dropna().unique())-{"F","M"},
            "address":set(df["address"].dropna().unique())-{"U","R"}
        }
    found=False
    for col,vals in checks.items():
        if vals:
            print(col,":",", ".join(map(str,vals)))
            found=True
    if not found:
        print("None")
print("\n"+"="*55)
print("DATA QUALITY COMPARISON")
print("="*55)
result=[]
for name,df in data.items():
    result.append([name,len(df),len(df.columns),int(df.isnull().sum().sum()),int(df.duplicated().sum())])
print(pd.DataFrame(result,columns=["Dataset","Records","Attributes","Missing","Duplicates"]).to_string(index=False))
