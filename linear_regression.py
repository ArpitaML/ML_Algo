#description
"The code's overall purpose is to showcase data preprocessing, visualization, manual data splitting, and training a simple machine learning model for diabetes prediction using Logistic Regression.Finally evaluate the model's accuracy"

#Importing Libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


#Loading and Examining the Data
diabetesDF = pd.read_csv('diabetes.csv')
print("DIABETES DATAFRAME: \n", diabetesDF.head())
print("FILE INFORMATION: \n", diabetesDF.info())


#Visualizing Data Correlations
corr = diabetesDF.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()


#Preparing Data for Machine Learning
x = diabetesDF.drop('Outcome', axis=1)
y = diabetesDF['Outcome']


#Manually Splitting Data
train_indices = range(0, int(len(x) * 0.7))
print("The train indices: \n", train_indices)				#70% train set
test_indices = range(int(len(x) * 0.7), len(x))
print("The test indices: \n", test_indices)				#30% test set
x_train, y_train = x.iloc[train_indices], y.iloc[train_indices]
x_test, y_test = x.iloc[test_indices], y.iloc[test_indices]


#Creating and Training a Machine Learning Model
diabetesCheck = LogisticRegression()
diabetesCheck.fit(x_train, y_train)


#Evaluating Model Accuracy
accuracy = diabetesCheck.score(x_test, y_test)
print("accuracy = ",accuracy * 100,"%")

#Result
"""accuracy =  78.78787878787878 %"""

