from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()


features = iris.data
label = iris.target

clf = KNeighborsClassifier()

clf.fit(features, label)

pred = clf.predict([[3.2, 12.3, 7.6, 9.8]])

if pred == 0:
    print("Setosa")
elif pred == 1:
    print("Versicolour")
elif pred == 2:
    print("Virginica")
else:
    print("Sorry!")


# - Iris-Setosa
# - Iris-Versicolour
# - Iris-Virginica