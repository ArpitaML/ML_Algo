#description
"""The provided code is a Python script that demonstrates the creation, training, evaluation, and saving of a Random Forest classifier for predicting heart disease based on a dataset."""
# Importing the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Load the dataset
df = pd.read_csv('heart_v2.csv')                                  #loading 'heart_v2.csv' into a Pandas DataFrame 'df'

# Display the first few rows of the dataset
print("The First Few Rows Of The Dataset \n", df.head())          #initial glimpse of the data

# Count plot of heart disease patients
sns.countplot(data=df, x='heart disease')
plt.title('Value counts of heart disease patients')
plt.xticks([0, 1], ['No Disease', 'Disease'])
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.show()                                                        #distribution of heart disease patients (0 for No Disease, 1 for Disease)

# Separating features and target variable
X = df.drop('heart disease', axis=1)                                                                                   # all columns except the 'heart disease' column
y = df['heart disease']                                                                                                # contains the 'heart disease' column.

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)                              # 70% of the data into training set with fixed random state.

# Creating a Random Forest classifier 
classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=2, n_estimators=100, oob_score=True)       # classifier with specified parameters
classifier_rf.fit(X_train, y_train)                                                                                     # The model trained on training data

# Defining the parameter grid for hyperparameter tuning
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100, 200],
    'n_estimators': [10, 25, 30, 50, 100, 200]
}                                                                                                                       # params dictionary contains different values for hyperparameters to be used for hyperparameter tuning
# Instantiate the Random Forest model
rf = RandomForestClassifier(random_state=42, n_jobs=-1)                                                                 # instance of Random Forest classifier (rf) that will be used for hyperparameter tuning.

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=params, cv=4, n_jobs=-1, verbose=1, scoring="accuracy")              #to perform hyperparameter tuning

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)                                                                                        #find the best set of hyperparameters

# Print the best score from grid search
print("Best Score:", grid_search.best_score_)                                                                             # best cross-validated score achieved

# Get the best estimator from grid search
rf_best = grid_search.best_estimator_
print("Best Estimator:", rf_best)

# Convert the Index object to a list
feature_names_list = list(X.columns)

# Visualize each tree from the Random Forest one by one
for i, tree in enumerate(rf_best.estimators_):                                                                  # This loop iterates over the trees and displays them along with feature names and class names.
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=feature_names_list, class_names=['Disease', 'No Disease'], filled=True)
    plt.title(f"Tree {i + 1}")
    plt.show()
    
# Model Evaluation
y_pred = rf_best.predict(X_test)                                                                                # evaluating the model's performance on the test set
conf_matrix = confusion_matrix(y_test, y_pred)                                                                  # calculating the confusion matrix
print("Confusion Matrix:\n", conf_matrix)
class_report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])                    # generating a classification report
print("Classification Report:\n", class_report)

# Feature Importance
importances = rf_best.feature_importances_                                                                      # feature importances calculated as the mean and standard deviation of accumulation of the impurity decrease within each tree. Each score is calculated based on how much splitting a particular feature reduces impurity (usually Gini impurity or entropy) within each decision tree in the forest
sorted_idx = importances.argsort()
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [feature_names_list[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance for Random Forest")
plt.show()

# Save the trained model
model_filename = "rf_model.pkl"                                                          # saved model can be used for making predictions without needing to retrain the model
joblib.dump(rf_best, model_filename)

# Load the model
loaded_model = joblib.load(model_filename)                                               # can use this variable to perform predictions just like you would with the original trained model

# Example data for prediction
sample_data = X_test.iloc[0]                                                             # Use the first row from the test set

# Predict using the loaded model
loaded_prediction = loaded_model.predict([sample_data])
print("Loaded Model Predicted Class:", loaded_prediction)                                # predicted class label (0 or 1) for the example data point using the loaded model.

"""
RESULT (only printables)

The First Few Rows Of The Dataset 
    age  sex   BP  cholestrol  heart disease
0   70    1  130         322              1
1   67    0  115         564              0
2   57    1  124         261              1
3   64    1  128         263              0
4   74    0  120         269              0

Best Score: 0.6985815602836879
Best Estimator: RandomForestClassifier(max_depth=5, min_samples_leaf=10, n_estimators=10, n_jobs=-1, random_state=42)

Confusion Matrix:
 [[33 16]
 [14 18]]
Classification Report: 
               precision    recall  f1-score   support

  No Disease       0.70      0.67      0.69        49
     Disease       0.53      0.56      0.55        32

    accuracy                           0.63        81
   macro avg       0.62      0.62      0.62        81
weighted avg       0.63      0.63      0.63        81

"""
