import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn .metrics import accuracy_score

#Loading the Diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('/Users/triakshacodes/Diabetes_Prediction/sample dataset/diabetes.csv')

#printing first 5 rows of the dataset
print(diabetes_dataset.head())

#Number of rows and columns in this dataset
print(diabetes_dataset.shape)

#getting the statistical measures of the data
print(diabetes_dataset.describe())

print(diabetes_dataset['Outcome'].value_counts())

print(diabetes_dataset.groupby('Outcome').mean())

#seperating data and labels
data_without_outcome = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']

#Data Standardization
scaler = StandardScaler()
scaler.fit(data_without_outcome)
standardized_data = scaler.transform(data_without_outcome)

X = standardized_data

#Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)

#Training the model
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train, Y_train)

#model evaluation
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy score of the training data = ", training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score of the testing data = ", testing_data_accuracy)

#making a predictive system
data = list(eval(input("Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age : ")))
data_array = np.asarray(data)
data_reshaped = data_array.reshape(1, -1)
std_data = scaler.transform(data_reshaped)

print(std_data)

prediction = classifier.predict(std_data)

if (prediction[0] == [0]):
    print("The person is not diabetic")
else:
    print("the person is diabetic")


