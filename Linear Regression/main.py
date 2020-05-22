import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes_data = datasets.load_diabetes()

# ['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename']
# print(diabetes_data.keys())
# print(diabetes_data.data) # This prints the entire data's arrays
# print(diabetes_data.DESCR)

# Below we're selecting one label and one feature
# This code gives the column second to diabetes_X and np converts it into numpy array (array of arrays)
# diabetes_X = diabetes_data.data[:, np.newaxis, 2]

diabetes_X = diabetes_data.data # This select all the features

# print(diabetes_X)

# Now we're doing train test splitting
diabetes_X_train = diabetes_X[:-20]     # Here we're selecting last 20 features
diabetes_X_test = diabetes_X[-20:]      # Here we're selecting first 20 fetures

diabetes_y_train = diabetes_data.target[:-20]  # The corresponding label for the X Train features
diabetes_y_test = diabetes_data.target[-20:]    # Same for the X test

model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_predicted = model.predict(diabetes_X_test)

print("Mean squared error: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))

# Printing the weights and intercepts w0 and w1, w2 up to n

# coef means the co-efficients this shows the theta tan theta which 941 (These are the w1, w2, w3)
print("Weights: ", model.coef_)

# This is the w0
print("Intercept: ", model.intercept_)

# The number of values it plots are 20 which we selected before
# plt.scatter(diabetes_X_test, diabetes_y_test)
# This plot the Linear Line for the Predicted values
# plt.plot(diabetes_X_test, diabetes_y_predicted)

# plt.show()
