# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Here the basic neural network model has been created with one input layer, one hidden layer and one output layer.The number of neurons(UNITS) in each layer varies the 1st input layer has 16 units and hidden layer has 8 units and output layer has one unit.

In this basic NN Model, we have used "relu" activation function in input and hidden layer, relu(RECTIFIED LINEAR UNIT) Activation function is a piece-wise linear function that will output the input directly if it is positive and zero if it is negative.



## Neural Network Model
![output](https://github.com/Hemapriya-2004/basic-nn-model/blob/main/k1.png)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
~~~
NAME:R.HEMAPRIYA
REG.NO:212221230036
~~~
```

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('mydata').sheet1
rows = worksheet.get_all_values()

dataset1 = pd.DataFrame(rows[1:], columns=rows[0])
dataset1 = dataset1.astype({'input':'float'})
dataset1 = dataset1.astype({'output':'float'})

dataset1.head()

X=dataset1[{'input'}].values
Y=dataset1[{'output'}].values
X

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33, random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

model = Sequential([
  Dense(8,activation = 'relu'),
  Dense(10,activation ='relu'),
  Dense(1)
])

model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,Y_train,epochs=2000)

## Plot the loss
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

## Evaluate the model
X_test1 = Scaler.transform(X_test)
model.evaluate(X_test1,Y_test)

# Prediction
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
model.predict(X_n1_1)

```
## Dataset Information

![output](https://github.com/Hemapriya-2004/basic-nn-model/blob/main/k2.png)


## OUTPUT

### Training Loss Vs Iteration Plot

![output](https://github.com/Hemapriya-2004/basic-nn-model/blob/main/k3.png)

### Test Data Root Mean Squared Error
![output](https://github.com/Hemapriya-2004/basic-nn-model/blob/main/k4.png)

### New Sample Data Prediction

![output](https://github.com/Hemapriya-2004/basic-nn-model/blob/main/k5.png)

## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully.
