from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris.DESCR)

# We're doing slicing, from data we're taking only one feature petal width
# Taking all rows and only the 4th column starts from 0
X = iris['data'][:, 3:]
# print(X)

'''We're looking for whether tha flower in iris verginica or not
that's why taking no 2 below'''

# Creating binary classifier and converting those true false to int
y = ((iris['target']) == 2).astype(np.int)
# print(y)

# Training a Logistic Regression Classifier Model
clf = LogisticRegression()
clf.fit(X, y)

example = clf.predict([[1.6]])
print(example)


# Using Matplotlib to plot

# Means taking 1000 points b/w 0 and 3
# and reshaping to a 1D array
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # plotting on x axis
print(X_new)

y_prob = clf.predict_proba(X_new)  # plotting on y axis

plt.plot(X_new, y_prob[:, 1], "g-", label="virginica")
plt.show()