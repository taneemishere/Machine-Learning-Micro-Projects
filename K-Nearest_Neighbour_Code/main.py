from sklearn import datasets    # for data sets
from sklearn.neighbors import KNeighborsClassifier  # for classifier model

# Import the data
iris = datasets.load_iris()

# Printing the data description
# print(iris.DESCR)

# Assigning features and label
features = iris.data
label = iris.target

# print(features[0], label[0])

# Building a K Nearest Neighbour Classifier Model
clf = KNeighborsClassifier()

# Training the Classifier Model
clf.fit(features, label)

# Predicting the values by model
pred = clf.predict([[3.5, 5.3, 2.6, 1.8]])

# printing the predicted value
print(pred)
