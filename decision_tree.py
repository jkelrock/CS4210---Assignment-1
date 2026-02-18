# -------------------------------------------------------------------------
# AUTHOR: Sofia Truong
# FILENAME: decision_tree
# SPECIFICATION: ID3 Algorithm that takes a dataset from a .csv file, and creates a decision tree using entropy and IG.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 30 minutes.
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

db = []
X = []
Y = []

# reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

# encode the original categorical training features into numbers and add to the 4D array X.
# --> add your Python code here
age_dict = {
    'Young': 0,
    'Prepresbyopic': 1,
    'Presbyopic': 2,
}
spectacle_dict = {
    'Myope': 0,
    'Hypermetrope': 1
}
astigmatism_dict = {
    'No': 0,
    'Yes': 1
}
tear_dict = {
    'Reduced': 0,
    'Normal': 1
}

for row in db:
    X.append([
        age_dict[row[0]],
        spectacle_dict[row[1]],
        astigmatism_dict[row[2]],
        tear_dict[row[3]]
    ])

# encode the original categorical training classes into numbers and add to the vector Y.
# --> addd your Python code here
contact_lens_dict = {
    'Yes': 0,
    'No': 1
}

for row in db:
    Y.append(contact_lens_dict[row[4]])

# fitting the depth-2 decision tree to the data using entropy as your impurity measure
# --> addd your Python code here

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf = clf.fit(X, Y)

# plotting decision tree
tree.plot_tree(clf,
               feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'],
               class_names=['Yes', 'No'],
               filled=True,
               rounded=True)

plt.show()

