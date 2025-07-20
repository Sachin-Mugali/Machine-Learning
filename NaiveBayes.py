import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  #to take data as train and spit
from sklearn.metrics import accuracy_score, classification_report #precision,accuracry score
from  sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB


#download dataset
data = pd.read_csv("program6.csv")
print(data.head())
print("dataset downloaded")

#split the data
x = data.drop('Outcome',axis = 1)
y=data['Outcome']
y = np.array(data.iloc[:,-1])

#split data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#initialise naive bayes classify
nb_classify = GaussianNB()

#train model using gaussian
nb_classify.fit(x_train, y_train)
y_pred = nb_classify.predict(x_test)

#Evaluate Model
accuracy=accuracy_score(y_test, y_pred)
print("accuracy value",accuracy)

print("classification report")
print(classification_report(y_test, y_pred))



