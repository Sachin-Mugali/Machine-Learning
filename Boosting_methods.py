import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
#iris = load_iris()
#x = iris.data
#y = iris.target

data = pd.read_csv("program6.csv")   #load the dataset of its name
x = data.drop('Outcome',axis=1)
y = data['Outcome']
#y = np.array(data.iloc[:,-1])

#split data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=42)

#ADAboost Classifier
ada = AdaBoostClassifier(n_estimators=50,random_state=42)
 
#train the model
ada.fit(x_train, y_train)
ada_pred = ada.predict(x_test)

#x_GradientBoosting classifier
gb = GradientBoostingClassifier(n_estimators=50,random_state=42)
gb.fit(x_train, y_train)
gb_pred = gb.predict(x_test)

#result ada boost classifier
print("ADAboost")
print("Accuracy:",accuracy_score(y_test, ada_pred))
print("Classfication Report:")
print(classification_report(y_test, ada_pred))

print("GradientBoosting classifier")
print("Accuracy:",accuracy_score(y_test, gb_pred))
print("Classfication Report:")
print(classification_report(y_test, gb_pred))

#if xgb classifier not working install pip install XGBoost
#XGBoost Classifier
xgb  = XGBClassifier(use_label_encounter=False)
xgb.fit(x_train, y_train)
xgb_pred=xgb.predict(x_test)

print("XGBBoost classifier")
print("Accuracy:",accuracy_score(y_test, xgb_pred))
print("Classfication Report:")
print(classification_report(y_test, xgb_pred))

#plot graph
plt.figure(figsize=(10,6))
plt.bar(data.BMI,gb.feature_importances_)
plt.title("feature VS Gradient")
plt.tight_layout()
plt.show()
