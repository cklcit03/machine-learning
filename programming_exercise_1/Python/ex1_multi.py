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
# Problem: Predict housing prices given sizes/bedrooms of various houses
# Linear regression with multiple variables
from matplotlib import pyplot
import numpy

class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def feature_normalize(X):
    """ Performs feature normalization.

    Args:
      X: Matrix of features.

    Returns:
      returnList: List of three objects.
                  x_normalized: Matrix of normalized features.
                  mu_vec: Vector of mean values of features.
                  sigma_vec: Vector of standard deviations of features.

    Raises:
      An error occurs if the number of features is 0.
      An error occurs if the number of training examples is 0.
    """
    num_ex = X.shape[0]
    if num_ex == 0: raise Error('num_ex = 0')
    num_features = X.shape[1]
    if num_features == 0: raise Error('num_features = 0')
    x_normalized = numpy.zeros((num_ex, num_features))
    mu_vec = numpy.mean(X, axis=0)
    sigma_vec = numpy.std(X, axis=0)
    for index in range(0, num_ex):
        x_normalized[index] = numpy.divide(numpy.subtract(X[index, :], mu_vec),
                                           sigma_vec)
    returnList = {'x_normalized': x_normalized, 'mu_vec': mu_vec,
                  'sigma_vec': sigma_vec}
    return returnList


def compute_cost_multi(X, y, theta):
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
    num_ex = y.shape[0]
    if num_ex == 0: raise Error('num_ex = 0')
    diff_vec = numpy.subtract(numpy.dot(X, theta), y)
    diff_vec_sq = numpy.multiply(diff_vec, diff_vec)
    j_theta = (numpy.sum(diff_vec_sq, axis=0))/(2*num_ex)
    return numpy.asscalar(j_theta)


def gradient_descent_multi(X, y, theta, alpha, numiters):
    """ Runs gradient descent.

    Args:
      X: Matrix of features.
      y: Vector of labels.
      theta: Vector of parameters for linear regression.
      alpha: Learning rate for gradient descent.
      numiters: Number of iterations for gradient descent.

    Returns:
      returnList: List of two objects.
                  theta: Updated vector of parameters for linear regression.
                  j_history: Vector of linear regression cost at each iteration.

    Raises:
      An error occurs if the number of features is 0.
      An error occurs if the number of iterations is 0.
      An error occurs if the number of training examples is 0.
    """
    if numiters <= 0: raise Error('numiters <= 0')
    num_ex = y.shape[0]
    if num_ex == 0: raise Error('num_ex = 0')
    num_features = X.shape[1]
    if num_features == 0: raise Error('num_features = 0')
    j_theta_array = numpy.zeros((numiters, 1))
    for theta_index in range(0, numiters):
        diff_vec = numpy.subtract(numpy.dot(X, theta), y)
        diff_vec_times_X = numpy.multiply(diff_vec,
                                          numpy.reshape(X[:, 0],
                                                        (num_ex, 1)))
        for feat in range(1, num_features):
            diff_vec_times_X = numpy.c_[diff_vec_times_X,
                                        numpy.multiply(diff_vec,
                                                       numpy.reshape(X[:, feat],
                                                                     (num_ex,
                                                                      1)))]
        theta_new = numpy.subtract(theta, alpha*(1/num_ex)*
                                   numpy.reshape(numpy.sum(diff_vec_times_X,
                                                           axis=0),
                                                 (num_features, 1)))
        j_theta_array[theta_index] = compute_cost_multi(X, y, theta_new)
        theta = theta_new
    returnList = {'theta': theta, 'j_history': j_theta_array}
    return returnList


def normal_eqn(X, y):
    """ Computes normal equations.

    Args:
      X: Matrix of features.
      y: Vector of labels.

    Returns:
      theta: Closed-form solution to linear regression.
    """
    theta = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(numpy.transpose(X),
                                                            X)),
                                numpy.transpose(X)), y)
    return theta


def main():
    """ Main function
    """
    print("Loading data ...")
    housing_data = numpy.genfromtxt("../housingData.txt", delimiter=",")
    x_mat = numpy.c_[housing_data[:, 0], housing_data[:, 1]]
    num_ex = housing_data.shape[0]
    y_vec = numpy.reshape(housing_data[:, 2], (num_ex, 1))
    print("First 10 examples from the dataset:")
    for training_ex_index in range(0, 10):
        print(" x = %s, y = %s" %
              (numpy.array_str(x_mat[training_ex_index, :].astype(int)),
               numpy.array_str(y_vec[training_ex_index, :].astype(int))))
    input("Program paused. Press enter to continue.")
    print("")

    # Perform feature normalization
    print("Normalizing Features ...")
    feature_normalize_list = feature_normalize(x_mat)
    x_mat_normalized = feature_normalize_list['x_normalized']
    mu_vec = feature_normalize_list['mu_vec']
    sigma_vec = feature_normalize_list['sigma_vec']
    ones_vec = numpy.ones((num_ex, 1))
    x_mat_aug = numpy.c_[ones_vec, x_mat]
    x_mat_normalized_aug = numpy.c_[ones_vec, x_mat_normalized]
    theta_vec = numpy.zeros((3, 1))
    iterations = 400
    alpha = 0.1

    # Run gradient descent
    print("Running gradient descent ...")
    gradient_descent_multi_list = gradient_descent_multi(x_mat_normalized_aug,
                                                         y_vec, theta_vec,
                                                         alpha, iterations)
    theta_final = gradient_descent_multi_list['theta']
    j_history = gradient_descent_multi_list['j_history']
    pyplot.plot(numpy.arange(j_history.shape[0]), j_history, 'b-',
                markersize=18)
    pyplot.ylabel('Cost J', fontsize=18)
    pyplot.xlabel('Number of iterations', fontsize=18)
    pyplot.show()
    print("Theta computed from gradient descent:")
    print("%s\n" % numpy.array_str(numpy.round(theta_final, 6)))

    # Predict price for a 1650 square-foot house with 3 bedrooms
    x_mat_norm_1 = numpy.reshape(numpy.divide(numpy.subtract(numpy.array([1650,
                                                                          3]),
                                                             mu_vec),
                                              sigma_vec), (1, 2))
    x_mat_norm_1_aug = numpy.c_[numpy.ones((1, 1)), x_mat_norm_1]
    pred_price_1 = numpy.asscalar(numpy.dot(x_mat_norm_1_aug, theta_final))
    print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%.6f" %
          pred_price_1)
    input("Program paused. Press enter to continue.")
    print("")

    # Solve normal equations
    print("Solving with normal equations...")
    theta_normal = normal_eqn(x_mat_aug, y_vec)
    print("Theta computed from the normal equations:")
    print("%s\n" % numpy.array_str(numpy.round(theta_normal, 6)))

    # Use normal equations to predict price for a 1650 square-foot house with 3
    # bedrooms
    x_mat_2 = numpy.array([1, 1650, 3])
    pred_price_2 = numpy.asscalar(numpy.dot(x_mat_2, theta_normal))
    print("Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%.6f" %
          pred_price_2)

# Call main function
if __name__ == "__main__":
    main()
