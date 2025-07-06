
#ml lab linear regression 

#https://www.statlearning.com/resources-first-edition  source file
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression #importing linear regression from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import seaborn as sns

#load dataset
data = pd.read_csv("C:/Users/admin/Desktop/mca067/Advertising.csv")
print(data.head())
print("first five lines printed")
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

#explore the data
print(data.head())
print(data.describe())
print("all details printed")
sns.pairplot(data,x_vars=['TV'],y_vars='sales',height=3,aspect=1,kind='scatter')#can delete height aspect kind when error comes 
plt.show()

#Features target
x=data[['TV']] #independent 
y=data['sales'] #dependent 

#training dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#create the train model
model = LinearRegression()
model.fit(x_train, y_train)
#predict values
y_pred = model.predict(x_test)
#evaluate
print("Coefficients:",model.coef_[0])
print("Intercept:",model.intercept_)
print("Mean square error:",mean_squared_error( y_test, y_pred,))
print("R2 score:",r2_score(y_test, y_pred))

plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, y_pred, color = 'red')
plt.title('TV advertising vs sales')
plt.xlabel('TV advertising spen vs sales')
plt.ylabel('sales($k)')
plt.show()

#for multiple regression
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, y_pred, color = 'red')
plt.title('TV advertising vs sales')
plt.xlabel('TV advertising spen vs sales')
plt.ylabel('sales($k)')
plt.show()
