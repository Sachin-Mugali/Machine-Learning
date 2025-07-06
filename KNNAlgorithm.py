import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#load the dataset from the iris.csv file

data = pd.read_csv("Iris.csv")
print(data)
print("downloaded successfully")

X = data.drop('species',axis = 1)  #X capital and y is small letter
y = data['species']


#split the dataset into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30)


#feature scaling(standardising the data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #after scaling it increased  decimal to more than 1 value
X_test_scaled = scaler.transform(X_test)


#initialize the classifiers with k=3
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train_scaled,y_train)


#make predictions on test datas
y_pred = classifier.predict(X_test_scaled)


#Evaluate the model and training data using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy of the KNN model:{accuracy * 100:.2f}%")
#or print(f"accuracy of the KNN model:",accuracy * 100)#2f means upto decimal points

#optionally print the predicted vs actual labels
print(f"predicted labels:{y_pred}")
print(f"actual labels:{y_test.values}")
