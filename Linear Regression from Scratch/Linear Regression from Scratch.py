from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random


style.use('fivethirtyeight')

#xs = [1, 2, 3, 4, 5, 6]
#ys = [5, 4, 6, 4, 5, 7]

# plt.plot(xs, ys)
# plt.scatter(xs, ys)
# plt.show()

#xs = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
#ys = np.array([5, 4, 6, 5, 6, 7], dtype = np.float64)


def create_dataset(number_of_datapoint, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(number_of_datapoint):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]    # we can give the len(number_of_datapoint)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):

    x_mean = mean(xs)
    y_mean = mean(ys)

    xy_means = (x_mean*y_mean)
    mean_of_xy = mean(xs*ys)

    x_mean_sq = x_mean*x_mean
    x_sq_mean = mean(xs*xs)

    m = ((xy_means - mean_of_xy)/ (x_mean_sq - x_sq_mean))
    b = y_mean - m*x_mean

    
    return m, b


def sqaured_error(ys_original, ys_line):
    return sum((ys_line-ys_original)**2)


def coefficient_of_determination(ys_original, ys_line):
    y_mean_line = [mean(ys_original) for y in ys_original]
    squared_error_of_regression_line = sqaured_error(ys_original, ys_line)
    sqaured_error_y_mean_line = sqaured_error(ys_original, y_mean_line)

    return 1 - (squared_error_of_regression_line / sqaured_error_y_mean_line)


xs, ys = create_dataset(40, 40, 2, correlation='neg')

     
m, b = best_fit_slope_and_intercept(xs, ys)
#print(m, b)


#for x in xs:
#    regression_line.append(m*x)+b)


regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = m*predict_x+b

# Coefficient of determination
r_sqaured = coefficient_of_determination(ys, regression_line)
print(r_sqaured)


plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.scatter(predict_x, predict_y, s=100, color='g')
plt.show()










