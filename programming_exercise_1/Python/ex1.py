# Copyright (C) 2014  Caleb Lo
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Machine Learning
# Programming Exercise 1: Linear Regression
# Problem: Predict profits for a food truck given data for profits/populations
# of various cities
# Linear regression with one variable
from matplotlib import cm
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def plot_data(x, y):
    """ Plots data.

    Args:
      x: X-values of data to be plotted.
      y: Y-values of data to be plotted.

    Returns:
      None.
    """
    pyplot.plot(x, y, 'rx', markersize=18)
    pyplot.ylabel('Profit in $10,0000s', fontsize=18)
    pyplot.xlabel('Population of City in 10,000s', fontsize=18)
    return None


def compute_cost(X, y, theta):
    """ Computes cost function J(\theta).

    Args:
      X: Matrix of features.
      y: Vector of labels.
      theta: Vector of parameters for linear regression.

    Returns:
      j_theta: Linear regression cost.

    Raises:
      An error occurs if the number of training examples is 0.
    """
    num_train_ex = y.shape[0]
    if num_train_ex == 0: raise Error('num_train_ex = 0')
    diff_vec = numpy.subtract(numpy.dot(X, theta), y)
    diff_vec_sq = numpy.multiply(diff_vec, diff_vec)
    j_theta = (numpy.sum(diff_vec_sq, axis=0))/(2*num_train_ex)
    return numpy.asscalar(j_theta)


def gradient_descent(X, y, theta, alpha, numiters):
    """ Runs gradient descent.

    Args:
      X: Matrix of features.
      y: Vector of labels.
      theta: Vector of parameters for linear regression.
      alpha: Learning rate for gradient descent.
      numiters: Number of iterations for gradient descent.

    Returns:
      theta: Updated vector of parameters for linear regression.

    Raises:
      An error occurs if the number of iterations is 0.
      An error occurs if the number of training examples is 0.
    """
    if numiters <= 0: raise Error('numiters <= 0')
    num_train_ex = y.shape[0]
    if num_train_ex == 0: raise Error('num_train_ex = 0')
    j_theta_array = numpy.zeros((numiters, 1))
    for theta_index in range(0, numiters):
        diff_vec = numpy.subtract(numpy.dot(X, theta), y)
        diff_vec_times_X = numpy.c_[numpy.multiply(diff_vec,
                                                   numpy.reshape(X[:, 0],
                                                                 (num_train_ex,
                                                                  1))),
                                    numpy.multiply(diff_vec,
                                                   numpy.reshape(X[:, 1],
                                                                 (num_train_ex,
                                                                  1)))]
        theta_new = numpy.subtract(theta, alpha*(1/num_train_ex)*
                                   numpy.reshape(numpy.sum(diff_vec_times_X,
                                                           axis=0), (2, 1)))
        j_theta_array[theta_index] = compute_cost(X, y, theta_new)
        theta = theta_new
    return theta


def main():
    """ Main function
    """
    print("Plotting Data ...")
    food_truck_data = numpy.genfromtxt("../foodTruckData.txt", delimiter=",")
    return_code = plot_data(food_truck_data[:, 0], food_truck_data[:, 1])
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")
    print("Running Gradient Descent ...")
    num_train_ex = food_truck_data.shape[0]
    ones_vec = numpy.ones((num_train_ex, 1))
    x_mat = numpy.c_[ones_vec, food_truck_data[:, 0]]
    theta_vec = numpy.zeros((2, 1))
    y_vec = numpy.reshape(food_truck_data[:, 1], (num_train_ex, 1))

    # Compute initial cost
    init_cost = compute_cost(x_mat, y_vec, theta_vec)
    print("ans = %.3f" % init_cost)

    # Run gradient descent
    iterations = 1500
    alpha = 0.01
    theta_final = gradient_descent(x_mat, y_vec, theta_vec, alpha, iterations)
    print("Theta found by gradient descent: %s" %
          numpy.array_str(numpy.transpose(numpy.round(theta_final, 6))))
    return_code = plot_data(food_truck_data[:, 0], food_truck_data[:, 1])
    pyplot.hold(True)
    pyplot.plot(x_mat[:, 1], numpy.dot(x_mat, theta_final), 'b-', markersize=18)
    pyplot.legend(('Training data', 'Linear regression'), loc='lower right')
    pyplot.hold(False)
    pyplot.show()

    # Predict profit for population size of 35000
    pred_profit_1 = numpy.asscalar(numpy.dot(numpy.array([1, 3.5]),
                                             theta_final))
    pred_profit_1_scaled = 10000*pred_profit_1
    print("For population = 35,000, we predict a profit of %.6f" %
          pred_profit_1_scaled)

    # Predict profit for population size of 70000
    pred_profit_2 = numpy.asscalar(numpy.dot(numpy.array([1, 7.0]),
                                             theta_final))
    pred_profit_2_scaled = 10000*pred_profit_2
    print("For population = 70,000, we predict a profit of %.6f" %
          pred_profit_2_scaled)
    input("Program paused. Press enter to continue.")
    print("")
    print("Visualizing J(theta_0, theta_1) ...\n")
    theta_0_vals = numpy.linspace(-10, 10, num=100)
    theta_1_vals = numpy.linspace(-1, 4, num=100)
    j_vals = numpy.zeros((theta_0_vals.shape[0], theta_1_vals.shape[0]))
    for theta_0_index in range(0, theta_0_vals.shape[0]):
        for theta_1_index in range(0, theta_1_vals.shape[0]):
            t_vec = numpy.vstack((theta_0_vals[theta_0_index],
                                  theta_1_vals[theta_1_index]))
            j_vals[theta_0_index, theta_1_index] = compute_cost(x_mat, y_vec,
                                                                t_vec)
    j_vals_trans = numpy.transpose(j_vals)
    theta_0_vals_X, theta_1_vals_Y = numpy.meshgrid(theta_0_vals, theta_1_vals)
    j_vals_reshape = j_vals_trans.reshape(theta_0_vals_X.shape)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta_0_vals_X, theta_1_vals_Y, j_vals_reshape, rstride=1,
                    cstride=1, cmap=cm.jet)
    ax.set_ylabel(r'$\Theta_1$', fontsize=18)
    ax.set_xlabel(r'$\Theta_0$', fontsize=18)
    pyplot.show()
    pyplot.contour(theta_0_vals_X, theta_1_vals_Y, j_vals_reshape,
                   levels=numpy.logspace(-2, 3, 20))
    pyplot.ylabel(r'$\Theta_1$', fontsize=18)
    pyplot.xlabel(r'$\Theta_0$', fontsize=18)
    pyplot.hold(True)
    pyplot.plot(theta_final[0], theta_final[1], 'rx', markersize=18,
                markeredgewidth=3)
    pyplot.hold(False)
    pyplot.show()

# Call main function
if __name__ == "__main__":
    main()
