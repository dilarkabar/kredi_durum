import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbr

data=pd.read_csv("Top_12_German_Companies.csv")
def veriyi_tanıma(dataframe,col):
    print(dataframe.head())
    print("**********************************************************************")
    print(dataframe.tail())
    print("**********************************************************************")
    print(dataframe.info())
    print("**********************************************************************")
    print(dataframe.dtypes)
    print("**********************************************************************")
    print(dataframe.describe().T)
    print("**********************************************************************")
    print(dataframe[col].value_counts())
    print("**********************************************************************")
    print(dataframe.isnull().sum())
    print("**********************************************************************")
    x=dataframe[col].isnull().sum()
    print((x/len(dataframe))*100)

def veri_türü(dataframe,cat=10,car=20):
    cat_col=[col for col in dataframe.columns if dataframe[col].dtypes=="O"]
    num_cat=[col for col in dataframe.columns if dataframe[col].nunique()<cat and
           dataframe[col].dtypes!="O" ]
    cat_car=[col for col in dataframe.columns if dataframe[col].dtypes=="O" and
              dataframe[col].nunique()>car ]
    cat_col=cat_col+num_cat
    cat_col=[col for col in cat_col if col not in cat_car]

    num_col=[col for col in dataframe.columns if dataframe[col].dtypes!="O"]
    num_col =num_cat+num_col
    num_col=[col for col in num_col if col not in num_cat]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_col)}')
    print(f'num_cols: {len(num_col)}')
    print(f'cat_but_car: {len(cat_car)}')
    print(f'num_but_cat: {len(num_cat)}')
    return cat_col, num_col, cat_car
dr=cat_col, num_col, cat_car = veri_türü(data)
print(dr)



sbr.boxplot(x=data["Net Income"])
plt.show()

def outlire(dataframe,col,q1=0.25,q3=0.75):
    q1=dataframe[col].quantile(q1)
    q3=dataframe[col].quantile(q3)
    iqr=q3-q1
    lower_limit=q1-(1.5*iqr)
    high_limit=q3+(1.5*iqr)
    return lower_limit,high_limit
outlire(data, "Net Income")
dr=outlire(data, "Net Income")
print(dr)

def control_outlire(dataframe,col):
    lower_limit,high_limit=outlire(dataframe,col)
    dataframe.loc[dataframe[col]>high_limit,col]=high_limit
    dataframe.loc[dataframe[col]<lower_limit,col]=lower_limit
    return high_limit,lower_limit
control_outlire(data, "Net Income")
dr=control_outlire(data, "Net Income")
print(dr)

def chek_outlier(dataframe,col):
    lower_limit,high_limit=outlire(dataframe,col)
    if dataframe[(dataframe[col] > high_limit) | (dataframe[col] < lower_limit)].any(axis=None):
        return True
    else:
        return False
chek_outlier(data, "Net Income")


def baskılama_yöntemi(dataframe, variable):
    lower_limit, higher_limit = outlire(dataframe, variable)
    dataframe.loc[dataframe[variable] < lower_limit, variable] = lower_limit
    dataframe.loc[dataframe[variable] > higher_limit, variable] = higher_limit
    return dataframe

baskılama_yöntemi(data, "Net Income")
outliers_exist = chek_outlier(data, "Net Income")
print(outliers_exist)
"""
#şirketlerin debt to equity görleşştirme (2017-2024)
import matplotlib.dates as mdates
for company in data['Company'].unique():
    company_data = data[data['Company'] == company]
    
    plt.figure(figsize=(8, 5))
    plt.plot(company_data['Period'], company_data['Debt to Equity'], marker='o', label=company, color='r')

    plt.title(f'Debt-to-Equity Oranı - {company}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Debt to Equity', fontsize=12)
    plt.grid(True)
   
    plt.xticks(rotation=45, ha='right')
    
plt.tight_layout()
plt.show()
"""
#şirketlerin risk durumlarını belirleme 
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

veriyi_tanıma(data,"Debt to Equity")
data["stıtuan"]=np.where(data["Debt to Equity"] >= 1, "High_risk", "low_risk")
print(data)

b_col=[col for col in data.columns if data[col].dtype not in [int,float] and data[col].nunique()==2]
print(b_col)
def label_encoding(dataframe,col):
    Lblencoder=LabelEncoder()
    dataframe[col]=Lblencoder.fit_transform(dataframe[col])
    return dataframe

for col in b_col:
    data=label_encoding(data, col)
print(data.head)

data.to_excel("12cmpny page.xlsx",index=False)

#tahminleme
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
x=data[["ROA (%)","ROE (%)","Debt to Equity"]]
y=data["stıtuan"]
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,
                                               random_state=42)
mdl=LogisticRegression()
mdl.fit(x_train,y_train)
yprd = mdl.predict(x_test)
mdl2=classification_report(y_test, yprd)
print(mdl2)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, yprd)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mdl.classes_)
disp.plot(cmap='viridis')
plt.show()









