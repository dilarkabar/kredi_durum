#Kanser Tahmin Veri Seti
#Yaş : Hastanın yaşını temsil eden 20 ile 80 arasında değişen tamsayı değerler.
#Cinsiyet : Cinsiyeti temsil eden ikili değerler; 0, Erkeği ve 1, Kadını belirtir.
#BMI : Vücut Kitle İndeksini temsil eden, 15 ile 40 arasında değişen sürekli değerler.
#Sigara İçme : Sigara içme durumunu gösteren ikili değerler; burada 0 Hayır, 1 ise Evet anlamına gelir.
#GeneticRisk : Kansere yönelik genetik risk düzeylerini temsil eden kategorik değerler; 0 Düşük, 1 Orta ve 2 Yüksek anlamına gelir.
#Fiziksel Aktivite : Haftada fiziksel aktivitelere harcanan saat sayısını temsil eden, 0 ile 10 arasında değişen sürekli değerler.
#Alkol Alımı : Haftada tüketilen alkol birimi sayısını temsil eden, 0 ile 5 arasında değişen sürekli değerler.
#Kanser Geçmişi : Hastanın kişisel bir kanser geçmişi olup olmadığını gösteren ikili değerler; burada 0 Hayır, 1 ise Evet anlamına gelir.
#Teşhis : Kanser teşhis durumunu gösteren ikili değerler; burada 0, Kanser Yok'u ve 1, Kanser'i belirtir.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate


data=pd.read_csv("The_Cancer_data.csv")
print(data.head(10))
print("*********************************")
print(data.tail(10))
print("*********************************")
print(data.info())
print("*********************************")
print(data.describe().T)
print("*********************************")
x=data["Diagnosis"].value_counts()

print((x/len(data))*100)
sns.countplot(x="Diagnosis", data=data)
plt.show()
a=data.groupby("Diagnosis").agg({"PhysicalActivity":"mean"})
print(a)
#model tahmini
y = data["Diagnosis"]

X = data.drop(["Diagnosis"], axis=1)
log_model = LogisticRegression().fit(X, y)

b=log_model.intercept_
w=log_model.coef_
print(b)

y_pred = log_model.predict(X)

y_pred[0:10]
cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
print(cv_results['test_accuracy'].mean())


print(cv_results['test_precision'].mean())


print(cv_results['test_recall'].mean())


print(cv_results['test_f1'].mean())

print(cv_results['test_roc_auc'].mean())

cm=confusion_matrix(y,y_pred)
print(f"confusion_matrix:{cm}")

cs=classification_report(y,y_pred)
print(f"classification_report:{cs}")

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.xlabel("tahmin")
plt.ylabel("gerçek")
plt.title("CONFSION MATRİX",color="red")
plt.show()

# örnek
random_user = X.sample(1, random_state=45)
print(log_model.predict(random_user))