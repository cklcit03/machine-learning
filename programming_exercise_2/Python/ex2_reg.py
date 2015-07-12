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
# Programming Exercise 2: Logistic Regression
# Problem: Predict chances of acceptance for a microchip given data for 
# acceptance decisions and test scores of various microchips
from matplotlib import pyplot
from scipy.optimize import fmin_ncg
import numpy

class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def plot_data(x, y):
    """ Plots data.

    Args:
      x: Features to be plotted.
      y: Data labels.

    Returns:
      None.
    """
    positive_indices = numpy.where(y == 1)
    negative_indices = numpy.where(y == 0)
    pos = pyplot.scatter(x[positive_indices, 0], x[positive_indices, 1], s=80,
                         marker='+', color='k')
    pyplot.hold(True)
    neg = pyplot.scatter(x[negative_indices, 0], x[negative_indices, 1], s=80,
                         marker='s', color='y')
    pyplot.legend((pos, neg), ('y = 1', 'y = 0'), loc='lower right')
    pyplot.hold(False)
    pyplot.ylabel('Microchip Test 2', fontsize=18)
    pyplot.xlabel('Microchip Test 1', fontsize=18)
    return None


def plot_decision_boundary(x, y, theta):
    """ Plots decision boundary.

    Args:
      x: Features that have already been plotted.
      y: Data labels.
      theta: Parameter that determines shape of decision boundary.

    Returns:
      None.
    """
    plot_data(numpy.c_[x[:, 0], x[:, 1]], y)
    pyplot.hold(True)
    theta_0_vals = numpy.linspace(-1, 1.5, num=50)
    theta_1_vals = numpy.linspace(-1, 1.5, num=50)
    j_vals = numpy.zeros((theta_0_vals.shape[0], theta_1_vals.shape[0]))
    for theta_0_index in range(0, theta_0_vals.shape[0]):
        for theta_1_index in range(0, theta_1_vals.shape[0]):
            j_vals[theta_0_index, theta_1_index] = (
                numpy.dot(map_feature(theta_0_vals[theta_0_index],
                                      theta_1_vals[theta_1_index]), theta))
    j_vals_trans = numpy.transpose(j_vals)
    theta_0_vals_X, theta_1_vals_Y = numpy.meshgrid(theta_0_vals, theta_1_vals)
    j_vals_reshape = j_vals_trans.reshape(theta_0_vals_X.shape)
    pyplot.contour(theta_0_vals, theta_1_vals, j_vals_reshape, 1)
    pyplot.hold(False)
    return None


def map_feature(X1, X2):
    """ Adds polynomial features to training data.

    Args:
      X1: Vector of values for feature 1.
      X2: Vector of values for feature 2.

    Returns:
      aug_x_mat: Vector of mapped features.
    """
    degree = 6
    num_train_ex = numpy.c_[X1, X2].shape[0]
    aug_x_mat = numpy.ones((num_train_ex, 1))
    for deg_index_1 in range(1, degree+1):
        for deg_index_2 in range(0, deg_index_1+1):
            aug_x_mat = (
                numpy.c_[aug_x_mat,
                         numpy.multiply(numpy.power(X1,
                                                    (deg_index_1-deg_index_2)),
                                        numpy.power(X2, deg_index_2))])
    return aug_x_mat


def compute_sigmoid(z):
    """ Computes sigmoid function.

    Args:
      z: Can be a scalar, a vector or a matrix.

    Returns:
      sigmoid_z: Sigmoid function value.
    """
    sigmoid_z = 1/(1+numpy.exp(-z))
    return sigmoid_z


def compute_cost(theta, X, y, num_train_ex, lamb):
    """ Computes regularized cost function J(\theta).

    Args:
      theta: Vector of parameters for regularized logistic regression.
      X: Matrix of features.
      y: Vector of labels.
      num_train_ex: Number of training examples.
      lamb: Regularization parameter.

    Returns:
      j_theta_reg: Regularized logistic regression cost.

    Raises:
      An error occurs if the number of features is 0.
      An error occurs if the number of training examples is 0.
    """
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = X.shape[1]
    if num_features == 0: raise Error('num_features = 0')
    theta = numpy.reshape(theta, (num_features, 1), order='F')
    h_theta = compute_sigmoid(numpy.dot(X, theta))
    theta_squared = numpy.power(theta, 2)
    j_theta = (numpy.sum(numpy.subtract(numpy.multiply(-y, numpy.log(h_theta)),
                                        numpy.multiply((1-y),
                                                       numpy.log(1-h_theta))),
                         axis=0))/num_train_ex
    j_theta_reg = (
        j_theta+(lamb/(2*num_train_ex))*numpy.sum(theta_squared,
                                                  axis=0)-theta_squared[0])
    return j_theta_reg


def compute_gradient(theta, X, y, num_train_ex, lamb):
    """ Computes gradient of regularized cost function J(\theta).

    Args:
      theta: Vector of parameters for regularized logistic regression.
      X: Matrix of features.
      y: Vector of labels.
      num_train_ex: Number of training examples.
      lamb: Regularization parameter.

    Returns:
      grad_array_reg_flat: Vector of regularized logistic regression gradients
                           (one per feature).

    Raises:
      An error occurs if the number of features is 0.
      An error occurs if the number of training examples is 0.
    """
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = X.shape[1]
    if num_features == 0: raise Error('num_features = 0')
    theta = numpy.reshape(theta, (num_features, 1), order='F')
    h_theta = compute_sigmoid(numpy.dot(X, theta))
    grad_array = numpy.zeros((num_features, 1))
    grad_array_reg = numpy.zeros((num_features, 1))
    for grad_index in range(0, num_features):
        grad_term = numpy.multiply(numpy.reshape(X[:, grad_index],
                                                 (num_train_ex, 1)),
                                   numpy.subtract(h_theta, y))
        grad_array[grad_index] = (numpy.sum(grad_term, axis=0))/num_train_ex
        grad_array_reg[grad_index] = (
            grad_array[grad_index]+(lamb/num_train_ex)*theta[grad_index])
    grad_array_reg[0] = grad_array_reg[0]-(lamb/num_train_ex)*theta[0]
    grad_array_reg_flat = numpy.ndarray.flatten(grad_array_reg)
    return grad_array_reg_flat


def compute_cost_grad_list(X, y, theta, lamb):
    """ Aggregates computed cost and gradient.

    Args:
      X: Matrix of features.
      y: Vector of labels.
      theta: Vector of parameters for regularized logistic regression.
      lamb: Regularization parameter.

    Returns:
      return_list: List of two objects.
                   j_theta_reg: Updated vector of parameters for regularized 
                                logistic regression.
                   grad_array_reg: Updated vector of regularized logistic 
                                   regression gradients (one per feature).
    """
    num_features = X.shape[1]
    num_train_ex = y.shape[0]
    j_theta_reg = compute_cost(theta, X, y, num_train_ex, lamb)
    grad_array_reg_flat = compute_gradient(theta, X, y, num_train_ex, lamb)
    grad_array_reg = numpy.reshape(grad_array_reg_flat, (num_features, 1),
                                   order='F')
    return_list = {'j_theta_reg': j_theta_reg, 'grad_array_reg': grad_array_reg}
    return return_list


def label_prediction(X, theta):
    """ Performs label prediction on training data.

    Args:
      X: Matrix of features.
      theta: Vector of parameters for regularized logistic regression.

    Returns:
      p: Vector of predictions (one per example).

    Raises:
      An error occurs if the number of training examples is 0.
    """
    num_train_ex = X.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    sigmoid_array = compute_sigmoid(numpy.dot(X, theta))
    p = numpy.zeros((num_train_ex, 1))
    for train_index in range(0, num_train_ex):
        if (sigmoid_array[train_index] >= 0.5):
            p[train_index] = 1
        else:
            p[train_index] = 0
    return p


def main():
    """ Main function

    Raises:
      An error occurs if the number of training examples is 0.
    """
    micro_chip_data = numpy.genfromtxt("../microChipData.txt", delimiter=",")
    return_code = plot_data(numpy.c_[micro_chip_data[:, 0],
                                     micro_chip_data[:, 1]],
                            micro_chip_data[:, 2])
    pyplot.show()
    num_train_ex = micro_chip_data.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = micro_chip_data.shape[1]-1
    ones_vec = numpy.ones((num_train_ex, 1))
    x_mat = numpy.c_[micro_chip_data[:, 0], micro_chip_data[:, 1]]
    y_vec = numpy.reshape(micro_chip_data[:, 2], (num_train_ex, 1))

    # Add polynomial features to training data
    feature_x_mat = map_feature(x_mat[:, 0], x_mat[:, 1])
    theta_vec = numpy.zeros((feature_x_mat.shape[1], 1))

    # Compute initial cost and gradient
    lamb = 1
    init_compute_cost_list = compute_cost_grad_list(feature_x_mat, y_vec,
                                                    theta_vec, lamb)
    print("Cost at initial theta (zeros): %.6f" %
          init_compute_cost_list['j_theta_reg'])
    input("Program paused. Press enter to continue.")

    # Use fmin_ncg to solve for optimum theta and cost
    theta_vec_flat = numpy.ndarray.flatten(theta_vec)
    fmin_ncg_out = fmin_ncg(compute_cost, theta_vec_flat,
                            fprime=compute_gradient, args=(feature_x_mat, y_vec,
                                                           num_train_ex, lamb),
                            avextol=1e-10, epsilon=1e-10, maxiter=400,
                            full_output=1)
    theta_opt = numpy.reshape(fmin_ncg_out[0], (feature_x_mat.shape[1], 1),
                              order='F')

    # Plot decision boundary
    return_code = plot_decision_boundary(x_mat, y_vec, theta_opt)
    pyplot.show()

    # Compute accuracy on training set
    training_predict = label_prediction(feature_x_mat, theta_opt)
    num_train_match = 0
    for train_index in range(0, num_train_ex):
        if (training_predict[train_index] == y_vec[train_index]):
            num_train_match += 1
    print("Train Accuracy: %.6f" % (100*num_train_match/num_train_ex))

# Call main function
if __name__ == "__main__":
    main()
