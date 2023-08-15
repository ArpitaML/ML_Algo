# Import the necessary libraries
from sklearn.datasets import load_iris					                       # load the Iris dataset. The dataset is part of the scikit-learn library and is easily accessible through this function. 
from sklearn.tree import DecisionTreeClassifier				                       # create a decision tree classifier model
from sklearn.tree import export_graphviz				                       # export a decision tree in Graphviz format
from graphviz import Source						                       # for rendering Graphviz source code
from sklearn.model_selection import train_test_split                                           # split dataset into:training and test set
from sklearn.metrics import accuracy_score                                                     # evaluate the performance of your classifier

# Load the dataset
iris = load_iris()
X = iris.data[:, 2:]				               # petal length and width
y = iris.target				                       # target labels 

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=2)		     # splitting nodes criterion: entropy; max_depth= maximum level the tree can grow
tree_clf.fit(X_train, y_train)						             # fitting the model with the data X and labels y

# Predict using the trained model
y_pred = tree_clf.predict(X_test)					             # predicted labels for the samples in the test set

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)				              # model's performance on unseen data
print("Accuracy on Test Set:", accuracy)

# Plot the decision tree graph
export_graphviz(tree_clf, out_file="iris_tree.dot", 			           # source code is saved to a file named "iris_tree.dot"
                feature_names=iris.feature_names[2:], 
                class_names=iris.target_names, 
                rounded=True, filled=True)

with open("iris_tree.dot") as f:					            # opening and reading the content of the "iris_tree.dot"
    dot_graph = f.read()						            # content of the Graphviz source code stored in the dot_graph variable
 
Source(dot_graph)							            # render the decision tree visualization directly in the script.							

# Generate the PNG image from the DOT file using Graphviz command line tools
import subprocess
subprocess.call(['dot', '-Tpng', 'iris_tree.dot', '-o', 'iris_tree.png'])           #DOT (Graph Description Language)

"""
RESULT:
Accuracy on Test Set: 0.9666666666666667
iris_tree.png will be saved in the working directory
"""
