import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
# from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

# Read the CSV file into a DataFrame
der = pd.read_csv("kredipuan.csv")

slted=["Age","Annual_Income","Monthly_Inhand_Salary","Num_of_Delayed_Payment","Interest_Rate"]
df=der[slted]
# Define a function to describe data and perform initial exploratory analysis
def data_describe(data, col):
    print(data.head())
    print(">>>>>>>>>>>>")
    print(data.tail())
    print(">>>>>>>>>>>>")
    print(data.shape)
    print(">>>>>>>>>>>>")
    print(data.info())
    print(">>>>>>>>>>>>")
    print(data.describe().T)
    print(">>>>>>>>>>>>")
    x = data[col].value_counts()
    print(x)
    print(">>>>>>>>>>>>")
    oran = (x / len(data)) * 100
    print(oran)
    print(">>>>>>>>>>>>")
    print(data.isnull().sum())

# Call the data_describe function with the 'Interest_Rate' column
data_describe(df, "Age")
print(">>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# Feature Engineering
df['Age'] = df['Age'].str.replace('[_]', '', regex=True)
df['Age'] = df['Age'].astype(int)
df = df[(df['Age'] >= 10) & (df['Age'] <= 100)]

df['Annual_Income'] = df['Annual_Income'].str.replace(',', '', regex=True)
df['Annual_Income'] = df['Annual_Income'].str.replace('[_]', '', regex=True)
df['Annual_Income'] = df['Annual_Income'].astype(float)

df['Num_of_Delayed_Payment'].fillna(0, inplace=True)
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].str.replace('_', '')
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].astype(float)



print(">>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# Outlier Detection Function
def aykiri_deger(dataframe, col, q1=0.5, q3=0.95):
    q1 = dataframe[col].quantile(q1)
    q3 = dataframe[col].quantile(q3)
    ıqr = q3 - q1
    low_limit = q1 - (ıqr * 1.5)
    up_limit = q3 + (ıqr * 1.5)
    return low_limit, up_limit

# Check for Outliers Function
def kontrol_aykiri_deger(dataframe, col):
    low_limit, up_limit = aykiri_deger(dataframe, col)
    if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Categorize Column Names Function
def grap_colnames(dataframe, cat_th=10, car_th=20):
    cat_col = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() > car_th]
    cat_col = num_but_cat + cat_col
    cat_col = [col for col in cat_col if col not in num_but_cat]

    num_col = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_col = [col for col in num_col if col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_col)}')
    print(f'num_cols: {len(num_col)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_col, num_col, cat_but_car

# Replace Thresholds Function
def replace_thresholds(dataframe, col):
    low_limit, up_limit = aykiri_deger(dataframe, col)
    dataframe.loc[dataframe[col] < low_limit, col] = low_limit
    dataframe.loc[dataframe[col] > up_limit, col] = up_limit

# Call grap_colnames to get column names
cat_col, num_col, cat_but_car = grap_colnames(df)
num_col = [col for col in num_col if col not in ["ID", "Customer_ID"]]

# Check for Outliers in numerical columns and replace them
for col in num_col:
    print(col, kontrol_aykiri_deger(df, col))

for col in num_col:
    replace_thresholds(df, col)

for col in num_col:
    print(col, kontrol_aykiri_deger(df, col))

print(">>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# Missing Values Function
def missing_values(dataframe, na_name=False):
    nan_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    miss = dataframe[nan_cols].isnull().sum().sort_values(ascending=False)
    oran = (dataframe[nan_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([miss, np.round(oran, 2)], axis=1, keys=['miss', 'oran'])
    print(missing_df, end="\n")

    if na_name:
        return nan_cols

missing_values(df)
missing_values(df, True)

print(">>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# Fill Missing Values Function
def fill_missing_values(data):
    for column in df.columns:
        if df[column].dtype == np.number:
            df[column].fillna(df[column].mean(), inplace=True)
        
    return data

fill_missing_values(df)
df.isnull().sum().sort_values(ascending=False)

print(df)

print(">>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
# ÖZELLİK ÇIKARIMI [Annual_Income, Age, Monthly_Inhand_Salary, Num_of_Delayed_Payment ]
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] =100
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] >= 3000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 80
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 50

df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 90
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] >= 3000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 60
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000)  & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 40

df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 80
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] >= 3000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 50
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 30 

df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 90
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] >= 3000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 60
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 40

df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 80
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] >= 3000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 50
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000)  & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 30

df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 70
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] >= 3000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 40
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 20

df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 0
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] >= 3000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 0
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 0

df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 0
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] >= 3000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 0
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000)  & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 0

df.loc[(df['Annual_Income'] < 10000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 0
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] >= 3000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 0 
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] >= 3000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 0

df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 80
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >= 20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 50
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >= 20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 30
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 60
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 30
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 10
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) &(df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 70
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000) & (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 40
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 20

df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 1000)& (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 60
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 1000)& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 30
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 10
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 50
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 20
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 0
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 40
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 10
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >=20) & (df['Age'] <=60)&(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 0



df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 70
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 40
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 20
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 60
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 30
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 10
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) &(df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 37
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000) & (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 17
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 0


df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 1000)& (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 57
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 1000)& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 37
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 7
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 30
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 15
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000)  & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 8
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >60) & (df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 30
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >60) &(df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 23
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] >60) &(df['Age'] <=85)&(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 10

df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 29
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] =14
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 0
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] <20 ) & (df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 30
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] <20 ) & (df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 11
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000)  & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 6
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] <20 ) & (df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 22
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] <20 )  &(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 0
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 3000)&(df["Monthly_Inhand_Salary"] >= 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 0

df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 28
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 20
df.loc[(df['Annual_Income'] >= 50000) & (df['Age'] <20 ) & (df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 0
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 25
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 5
df.loc[(df['Annual_Income'] <50000) & (df['Annual_Income'] >= 10000)  & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 0
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]<3 ), 'Score'] = 0
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 1000)& (df["Num_of_Delayed_Payment"]>=3 )& (df["Num_of_Delayed_Payment"]<10 ), 'Score'] = 0
df.loc[(df['Annual_Income'] < 10000) & (df['Age'] <20 ) &(df["Monthly_Inhand_Salary"] < 1000) & (df["Num_of_Delayed_Payment"]>= 10), 'Score'] = 0


# Score için Situation gruplaması
def grouplms(dataframe, col):
    dataframe.loc[dataframe[col] >=50, "Situation"] = "credit is given"
    dataframe.loc[(dataframe[col] < 50) & (dataframe[col] >= 25), "Situation"] = "credit can be given"
    dataframe.loc[dataframe[col] < 25, "Situation"] = "credit cannot be given"
    return dataframe

df = grouplms(df, "Score")
print(df)

le = LabelEncoder()

df['Situation'] = le.fit_transform(df['Situation'])


    



# KAYDETMEK
df.to_excel("kredipuan_updated.xlsx", index=False)

#STANDARTLAŞTIRMA
scaler = StandardScaler() 
df[num_col] = scaler.fit_transform(df[num_col])

df.head()

# MODELLEME VE  # FEATURE IMPORTANCE
x=df.drop("Situation",axis=1) 
y=df[["Situation"]] 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=17)


rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Dizilerin boyutunu kontrol etme
if len(y_test) != len(y_pred):
    min_len = min(len(y_test), len(y_pred))
    y_test = y_test[:min_len]
    y_pred = y_pred[:min_len]
print(f"y_test boyutu: {len(y_test)}, y_pred boyutu: {len(y_pred)}")

#accuracy,recall,precision,f1-score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(accuracy, 2)}")

recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {round(recall, 3)}")

precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {round(precision, 2)}")

f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1: {round(f1, 2)}")

def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, x)



