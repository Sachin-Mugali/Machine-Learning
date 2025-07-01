Data Scraping 


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt 
import seaborn as sns

#Load dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target

#split into training and testing dataset 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#standardize features
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

#train svm model (using rbf kernal by default)
model = SVC(kernel='rbf',C=1.0,gamma='scale')
model.fit(x_train,y_train)

#predict and evaluate 
y_pred = model.predict(x_test)
print("Accuracy : ",accuracy_score(y_test, y_pred))
print("Classification Report:",classification_report(y_test, y_pred))

#plot graph
iris_df = sns.load_dataset('iris')
sns.pairplot(iris_df,hue='species')
plt.suptitle('iris dataset feature visualisation',y=1.02)
plt.show()


Output

Accuracy :  1.0
Classification Report:               precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
