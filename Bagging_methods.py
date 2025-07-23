
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#load dataset
iris=load_iris()
x=iris.data
y=iris.target
feature_names=iris.feature_names
target_names=iris.target_names

#split the data
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30,random_state=30)

#train the model (Random Forest Classifier)
clf=RandomForestClassifier(n_estimators=100, random_state=25)
clf.fit(x_train, y_train)

#Predictions and report
y_pred=clf.predict(x_test)

print("Classification Report")
print(classification_report(y_test, y_pred))

#Confusion matrix
cm=confusion_matrix(y_test, y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues")
plt.title("Confusion matrix")
plt.show()

#feature Importaces
importances=clf.feature_importances_
std=np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

forest_importances=pd.Series(importances,index=feature_names)
fig,ax=plt.subplots()
forest_importances.plot.bar(yerr=std,ax=ax)
ax.set_title("feature importances(main decrease in impurity)")
ax.set_ylabel("Importance")
plt.tight_layout()
plt.show()
