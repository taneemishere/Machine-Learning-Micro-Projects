from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print("Features are: ", cancer.feature_names)
# print("Labels are: ", cancer.target_names)

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print(x_train, y_train)

classes = ['malignant', 'benign']

# Support Vector Classifier of Support Vector Machine
# Here the C is the Soft Margin for this
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print("Accuracy of SVC: ", acc)


clf_two = KNeighborsClassifier(n_neighbors=9)
clf_two.fit(x_train, y_train)

y_pred_two = clf_two.predict(x_test)

acc_two = metrics.accuracy_score(y_test, y_pred_two)

print("Accuracy of KNN: ", acc_two)
