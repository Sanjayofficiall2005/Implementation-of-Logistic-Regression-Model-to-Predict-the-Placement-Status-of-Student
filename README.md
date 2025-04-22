# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data 
2. Print the placement data and salary data. 
3 .Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```python
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed By  : SANJAY M
RegisterNumber: 212223230187
```
```PYTHON
import pandas as pd
data=pd.read_csv("/content/Placement_Data (1).csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
#### TOP 5 ELEMENTS:
![image](https://github.com/user-attachments/assets/c51f6094-6914-4f50-81ec-01143a05243d)

![image](https://github.com/user-attachments/assets/08760214-4436-44ce-a084-b741198da937)

#### DATA DUPLICATE:
![WhatsApp Image 2025-04-07 at 15 42 16_c1aba188](https://github.com/user-attachments/assets/5cba6d4a-086e-43a9-b8de-5f00acd6480c)

#### Y_PREDICTION ARRAY:
![image](https://github.com/user-attachments/assets/39f5e632-208f-4726-a6ef-17085cbf2dad)


#### CONFUSIUON ARRAY:
![image](https://github.com/user-attachments/assets/f02bb96c-a0c0-41c1-8c87-2e9cbc95e6f4)


#### ACCURACY VALUE:
![image](https://github.com/user-attachments/assets/f1bbf528-f270-4971-a293-e7c9a40d604d)


#### CLASSIFICATION REPORT:
![image](https://github.com/user-attachments/assets/5fd9ff56-9d6c-47c7-9003-a8747c81f734)


#### PREDICTION:
![image](https://github.com/user-attachments/assets/e8650d24-04e0-4d3d-9bae-2090b79192cb)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
