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
# Programming Exercise 5: Regularized Linear Regression and Bias vs. Variance
# Problem: Predict amount of water flowing out of a dam given data for 
# change of water level in a reservoir
from matplotlib import pyplot
from scipy.optimize import fmin_bfgs
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
      return_list: List of three objects.
                   x_norm: Matrix of normalized features.
                   mu_vec: Vector of mean values of features.
                   sigma_vec: Vector of standard deviations of features.

    Raises:
      An error occurs if the number of training examples is 0.
      An error occurs if the number of features is 0.
    """
    num_train_ex = X.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = X.shape[1]
    if (num_features == 0): raise Error('num_features = 0')
    x_norm = numpy.zeros((num_train_ex, num_features))
    mu_vec = numpy.mean(X, axis=0)

    # Note that standard deviation for numpy uses a denominator of n
    # Standard deviation for R and Octave uses a denominator of n-1
    sigma_vec = numpy.std(X, axis=0, dtype=numpy.float32)
    for index in range(0, num_train_ex):
        x_norm[index] = numpy.divide(numpy.subtract(X[index, :], mu_vec),
                                     sigma_vec)
    return_list = {'x_norm': x_norm, 'mu_vec': mu_vec, 'sigma_vec': sigma_vec}
    return return_list


def compute_cost(theta, X, y, lamb):
    """ Computes regularized cost function J(\theta).

    Args:
      theta: Vector of parameters for regularized linear regression.
      X: Matrix of features.
      y: Vector of labels.
      lamb: Regularization parameter.

    Returns:
      j_theta_reg: Regularized linear regression cost.

    Raises:
      An error occurs if the number of training examples is 0.
      An error occurs if the number of features is 0.
    """
    num_train_ex = y.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = X.shape[1]
    if (num_features == 0): raise Error('num_features = 0')
    theta = numpy.reshape(theta, (num_features, 1), order='F')
    diff_vec = numpy.subtract(numpy.dot(X, theta), y)
    diff_vec_sq = numpy.multiply(diff_vec, diff_vec)
    j_theta = (numpy.sum(diff_vec_sq, axis=0))/(2.0*num_train_ex)
    theta_squared = numpy.power(theta, 2)
    j_theta_reg = (
        j_theta+(lamb/(2.0*num_train_ex))*(numpy.sum(theta_squared,
                                                     axis=0)-theta_squared[0]))
    return j_theta_reg


def compute_gradient(theta, X, y, lamb):
    """ Computes gradient of regularized cost function J(\theta).

    Args:
      theta: Vector of parameters for regularized linear regression.
      X: Matrix of features.
      y: Vector of labels.
      lamb: Regularization parameter.

    Returns:
      grad_array_reg_flat: Vector of regularized linear regression gradients
                           (one per feature).

    Raises:
      An error occurs if the number of training examples is 0.
      An error occurs if the number of features is 0.
    """
    num_train_ex = y.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = X.shape[1]
    if (num_features == 0): raise Error('num_features = 0')
    theta = numpy.reshape(theta, (num_features, 1), order='F')
    h_theta = numpy.dot(X, theta)
    grad_array = numpy.zeros((num_features, 1))
    grad_array_reg = numpy.zeros((num_features, 1))
    for grad_index in range(0, num_features):
        grad_term = numpy.multiply(numpy.reshape(X[:, grad_index],
                                                 (num_train_ex, 1)),
                                   numpy.subtract(h_theta, y))
        grad_array[grad_index] = (1/num_train_ex)*numpy.sum(grad_term, axis=0)
        grad_array_reg[grad_index] = (
            grad_array[grad_index]+(lamb/num_train_ex)*theta[grad_index])
    grad_array_reg[0] = grad_array_reg[0]-(lamb/num_train_ex)*theta[0]
    grad_array_reg_flat = numpy.ndarray.flatten(grad_array_reg)
    return grad_array_reg_flat


def train_linear_reg(X, y, lamb):
    """ Trains linear regression.

    Args:
      X: Matrix of features.
      y: Vector of labels.
      lamb: Regularization parameter.

    Returns:
      theta_opt: Best set of parameters found by fmin_bfgs.

    Raises:
      An error occurs if the number of features is 0.
    """
    num_features = X.shape[1]
    if (num_features == 0): raise Error('num_features = 0')
    init_theta = numpy.ones((num_features, 1))
    init_theta_flat = numpy.ndarray.flatten(init_theta)
    fmin_bfgs_out = fmin_bfgs(compute_cost, init_theta_flat,
                              fprime=compute_gradient, args=(X, y, lamb),
                              maxiter=100, full_output=1)
    theta_opt = numpy.reshape(fmin_bfgs_out[0], (num_features, 1), order='F')
    return theta_opt


def learning_curve(X, y, x_val, y_val, lamb):
    """ Generates values for learning curve.

    Args:
      X: Matrix of training features.
      y: Vector of training labels.
      x_val: Matrix of cross-validation features.
      y_val: Vector of cross-validation labels.
      lamb: Regularization parameter.

    Returns:
      return_list: List of two objects.
                   error_train: Vector of regularized linear regression costs
                                for training data (one per example).
                   error_val: Vector of regularized linear regression costs for
                              cross-validation data (one per example).

    Raises:
      An error occurs if the number of training examples is 0.
    """
    num_train_ex = y.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    error_train = numpy.zeros((num_train_ex, 1))
    error_val = numpy.zeros((num_train_ex, 1))
    for ex_index in range(0, num_train_ex):
        x_sub_mat = X[0:(ex_index+1), :]
        y_sub_vec = y[0:(ex_index+1), :]
        train_theta_vec = train_linear_reg(x_sub_mat, y_sub_vec, 1)
        error_train[ex_index] = compute_cost(train_theta_vec, x_sub_mat,
                                             y_sub_vec, 0)
        error_val[ex_index] = compute_cost(train_theta_vec, x_val, y_val, 0)
    return_list = {'error_train': error_train, 'error_val': error_val}
    return return_list


def validation_curve(X, y, x_val, y_val):
    """ Generates values for validation curve.

    Args:
      X: Matrix of training features.
      y: Vector of training labels.
      x_val: Matrix of cross-validation features.
      y_val: Vector of cross-validation labels.

    Returns:
      return_list: List of three objects.
                   lambda_vec: Vector of regularization parameters.
                   error_train: Vector of regularized linear regression costs
                                for training data (one per regularization
                                parameter).
                   error_val: Vector of regularized linear regression costs for
                              cross-validation data (one per regularization
                              parameter).
    """
    lambda_vec = numpy.c_[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    num_lambda = lambda_vec.shape[1]
    lambda_vec = numpy.reshape(lambda_vec, (num_lambda, 1), order='F')
    error_train = numpy.zeros((num_lambda, 1))
    error_val = numpy.zeros((num_lambda, 1))
    for lambda_index in range(0, num_lambda):
        curr_lambda = lambda_vec[lambda_index]
        train_theta_vec = train_linear_reg(X, y, curr_lambda)
        error_train[lambda_index] = compute_cost(train_theta_vec, X, y, 0)
        error_val[lambda_index] = compute_cost(train_theta_vec, x_val, y_val, 0)
    return_list = {'lambda_vec': lambda_vec, 'error_train': error_train,
                   'error_val': error_val}
    return return_list


def poly_features(X, p):
    """ Performs feature mapping for polynomial regression.

    Args:
      X: Matrix of training features.
      p: Maximum degree of polynomial mapping.

    Returns:
      x_poly: Matrix of mapped features.

    Raises:
      An error occurs if the maximum degree of polynomial mapping is at most 0.
    """
    if (p <= 0): raise Error('p <= 0')
    x_poly = numpy.zeros((X.shape[0], p))
    for deg_index in range(0, p):
        x_poly[:, deg_index:(deg_index+1)] = numpy.power(X, deg_index+1)
    return x_poly


def plot_fit(min_x, max_x, mu, sigma, theta, p):
    """ Plots polynomial regression fit.

    Args:
      min_x: Lower bound for x-axis of plot.
      max_x: Upper bound for x-axis of plot.
      mu: Vector of mean values of mapped features.
      sigma: Vector of standard deviations of mapped features.
      theta: Vector of learned parameters for polynomial regression.
      p: Maximum degree of polynomial mapping.

    Returns:
      None.
    """
    x_seq = numpy.arange(min_x-15, max_x+25, 0.05)
    x_seq_vec = numpy.reshape(x_seq, (x_seq.size, 1), order='F')
    x_poly = poly_features(x_seq_vec, p)
    x_p_norm = numpy.zeros((x_poly.shape[0], p))
    for index in range(0, x_poly.shape[0]):
        x_p_norm[index:(index+1), :] = (
            numpy.divide(numpy.subtract(x_poly[index, :], mu), sigma))
    ones_vec = numpy.ones((x_poly.shape[0], 1))
    x_p_norm = numpy.c_[ones_vec, x_p_norm]
    pyplot.plot(x_seq_vec, numpy.dot(x_p_norm, theta), 'b-')
    return None


def main():
    """ Main function

    Raises:
      An error occurs if the number of training examples is 0.
      An error occurs if the number of cross-validation examples is 0.
      An error occurs if the number of test examples is 0.
    """
    print("Loading and Visualizing Data ...")
    water_train_data = numpy.genfromtxt("../waterTrainData.txt", delimiter=",")
    num_train_ex = water_train_data.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = water_train_data.shape[1]-1
    water_val_data = numpy.genfromtxt("../waterValData.txt", delimiter=",")
    num_val_ex = water_val_data.shape[0]
    if (num_val_ex == 0): raise Error('num_val_ex = 0')
    water_test_data = numpy.genfromtxt("../waterTestData.txt", delimiter=",")
    num_test_ex = water_test_data.shape[0]
    if (num_test_ex == 0): raise Error('num_test_ex = 0')

    # Plot training data
    ones_train_vec = numpy.ones((num_train_ex, 1))
    x_mat = water_train_data[:, 0:num_features]
    y_vec = water_train_data[:, num_features:(num_features+1)]
    pyplot.plot(x_mat, y_vec, 'rx', markersize=18)
    pyplot.ylabel('Water flowing out of the dam (y)', fontsize=18)
    pyplot.xlabel('Change in water level (x)', fontsize=18)
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Compute cost for regularized linear regression
    x_mat = numpy.c_[ones_train_vec, x_mat]
    theta_vec = numpy.ones((2, 1))
    init_cost = compute_cost(theta_vec, x_mat, y_vec, 1)
    print("Cost at theta = [1 ; 1]: %.6f" % init_cost)
    print("(this value should be about 303.993192)")
    input("Program paused. Press enter to continue.")
    print("")

    # Compute gradient for regularized linear regression
    init_gradient = compute_gradient(theta_vec, x_mat, y_vec, 1)
    print("Gradient at theta = [1 ; 1]: %s" %
          numpy.array_str(numpy.transpose(numpy.round(init_gradient, 6))))
    print("(this value should be about [-15.303016 598.250744])")
    input("Program paused. Press enter to continue.")
    print("")

    # Train linear regression
    lamb = 0
    train_theta_vec = train_linear_reg(x_mat, y_vec, lamb)

    # Plot fit over data
    pyplot.plot(x_mat[:, 1], y_vec, 'rx', markersize=18)
    pyplot.ylabel('Water flowing out of the dam (y)', fontsize=18)
    pyplot.xlabel('Change in water level (x)', fontsize=18)
    pyplot.hold(True)
    pyplot.plot(x_mat[:, 1], numpy.dot(x_mat, train_theta_vec), 'b-',
                markersize=18)
    pyplot.hold(False)
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Generate values for learning curve
    ones_val_vec = numpy.ones((num_val_ex, 1))
    x_val_mat = numpy.c_[ones_val_vec, water_val_data[:, 0:num_features]]
    y_val_vec = water_val_data[:, num_features:(num_features+1)]
    learning_curve_list = learning_curve(x_mat, y_vec, x_val_mat, y_val_vec,
                                         lamb)

    # Plot learning curve
    pyplot.plot(numpy.arange(1, num_train_ex+1),
                learning_curve_list['error_train'])
    pyplot.title('Learning curve for linear regression')
    pyplot.ylabel('Error', fontsize=18)
    pyplot.xlabel('Number of training examples', fontsize=18)
    pyplot.axis([0, 13, 0, 150])
    pyplot.hold(True)
    pyplot.plot(numpy.arange(1, num_train_ex+1),
                learning_curve_list['error_val'], color='g')
    pyplot.hold(False)
    pyplot.legend(('Train', 'Cross Validation'), loc='upper right')
    pyplot.show()
    print("")
    tab_head = ["# Training Examples", "Train Error", "Cross Validation Error"]
    print("\t\t".join(tab_head))
    for ex_index in range(0, num_train_ex):
        tab_line = [' ', ex_index+1,
                    learning_curve_list['error_train'][ex_index, ],
                    learning_curve_list['error_val'][ex_index, ]]
        print(*tab_line, sep='\t\t')
    input("Program paused. Press enter to continue.")
    print("")

    # Perform feature mapping for polynomial regression
    p = 8
    x_poly = poly_features(x_mat[:, 1:2], p)
    x_p_norm = feature_normalize(x_poly)
    x_p_norm['x_norm'] = numpy.c_[ones_train_vec, x_p_norm['x_norm']]
    x_test_mat = water_test_data[:, 0:num_features]
    x_test_poly = poly_features(x_test_mat[:, 0:1], p)
    x_test_poly_norm = numpy.zeros((num_test_ex, p))
    for index in range(0, num_test_ex):
        x_test_poly_norm[index] = (
            numpy.divide(numpy.subtract(x_test_poly[index, :],
                                        x_p_norm['mu_vec']),
                         x_p_norm['sigma_vec']))
    ones_test_vec = numpy.ones((num_test_ex, 1))
    x_test_poly_norm = numpy.c_[ones_test_vec, x_test_poly_norm]
    x_val_poly = poly_features(x_val_mat[:, 1:2], p)
    x_val_poly_norm = numpy.zeros((num_val_ex, p))
    for index in range(0, num_val_ex):
        x_val_poly_norm[index] = (
            numpy.divide(numpy.subtract(x_val_poly[index, :],
                                        x_p_norm['mu_vec']),
                         x_p_norm['sigma_vec']))
    x_val_poly_norm = numpy.c_[ones_val_vec, x_val_poly_norm]
    print("Normalized Training Example 1:")
    print("%s\n" %
          numpy.array_str(numpy.round(numpy.transpose(x_p_norm['x_norm'][0:1,
                                                                         :]),
                                      6)))
    input("Program paused. Press enter to continue.")
    print("")

    # Train polynomial regression
    lamb = 0
    train_theta_vec = train_linear_reg(x_p_norm['x_norm'], y_vec, lamb)

    # Plot fit over data
    pyplot.plot(x_mat[:, 1], y_vec, 'rx', markersize=18)
    pyplot.title('Polynomial Regression Fit (lambda = %f)' % lamb)
    pyplot.ylabel('Water flowing out of the dam (y)', fontsize=18)
    pyplot.xlabel('Change in water level (x)', fontsize=18)
    pyplot.axis([-100, 100, -80, 80])
    pyplot.hold(True)
    return_code = plot_fit(numpy.amin(x_mat[:, 1]), numpy.amax(x_mat[:, 1]),
                           x_p_norm['mu_vec'], x_p_norm['sigma_vec'],
                           train_theta_vec, p)
    pyplot.hold(False)
    pyplot.show()

    # Generate values for learning curve for polynomial regression
    learning_curve_list = learning_curve(x_p_norm['x_norm'], y_vec,
                                         x_val_poly_norm, y_val_vec, lamb)

    # Plot learning curve
    pyplot.plot(numpy.arange(1, num_train_ex+1),
                learning_curve_list['error_train'])
    pyplot.title('Polynomial Regression Learning Curve (lambda = %.6f)' % lamb)
    pyplot.ylabel('Error', fontsize=18)
    pyplot.xlabel('Number of training examples', fontsize=18)
    pyplot.axis([0, 13, 0, 100])
    pyplot.hold(True)
    pyplot.plot(numpy.arange(1, num_train_ex+1),
                learning_curve_list['error_val'], color='g')
    pyplot.hold(False)
    pyplot.legend(('Train', 'Cross Validation'), loc='upper right')
    pyplot.show()
    print("")
    print("Polynomial Regression (lambda = %.6f)" % lamb)
    print("")
    tab_head = ["# Training Examples", "Train Error", "Cross Validation Error"]
    print("\t\t".join(tab_head))
    for ex_index in range(0, num_train_ex):
        tab_line = [' ', ex_index+1,
                    learning_curve_list['error_train'][ex_index, ],
                    learning_curve_list['error_val'][ex_index, ]]
        print(*tab_line, sep='\t\t')
    input("Program paused. Press enter to continue.")
    print("")

    # Generate values for validation curve for polynomial regression
    validation_curve_list = validation_curve(x_p_norm['x_norm'], y_vec,
                                             x_val_poly_norm, y_val_vec)

    # Plot validation curve
    pyplot.plot(validation_curve_list['lambda_vec'],
                validation_curve_list['error_train'])
    pyplot.ylabel('Error', fontsize=18)
    pyplot.xlabel('lambda', fontsize=18)
    pyplot.hold(True)
    pyplot.plot(validation_curve_list['lambda_vec'],
                validation_curve_list['error_val'], color='g')
    pyplot.hold(False)
    pyplot.legend(('Train', 'Cross Validation'), loc='upper right')
    pyplot.show()
    print("")
    tab_head = ["lambda", "Train Error", "Cross Validation Error"]
    print("\t\t".join(tab_head))
    for lambda_index in range(0, validation_curve_list['lambda_vec'].shape[0]):
        tab_line = [validation_curve_list['lambda_vec'][lambda_index],
                    validation_curve_list['error_train'][lambda_index, ],
                    validation_curve_list['error_val'][lambda_index, ]]
        print(*tab_line, sep='\t\t')
    input("Program paused. Press enter to continue.")
    print("")

# Call main function
if __name__ == "__main__":
    main()
