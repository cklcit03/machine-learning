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
# Programming Exercise 3: (Multi-class) Logistic Regression and Neural Networks
# Problem: Predict label for a handwritten digit given data for 
# pixel values of various handwritten digits
from matplotlib import cm
from matplotlib import pyplot
from scipy.optimize import fmin_ncg
import numpy


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def display_data(X):
    """ Displays 2D data in a grid.

    Args:
      X: Matrix of 2D data that will be displayed using imshow().

    Returns:
      None.

    Raises:
      An error occurs if the number of rows is 0.
      An error occurs if the number of cols is 0.
    """
    num_rows = X.shape[0]
    if (num_rows == 0): raise Error('num_rows = 0')
    num_cols = X.shape[1]
    if (num_cols == 0): raise Error('num_cols = 0')
    example_width = (numpy.round(numpy.sqrt(num_cols))).astype(int)
    example_height = (num_cols/example_width)
    display_rows = (numpy.floor(numpy.sqrt(num_rows))).astype(int)
    display_cols = (numpy.ceil(num_rows/display_rows)).astype(int)
    pad = 1
    display_array = (-1)*numpy.ones((pad+display_rows*(example_height+pad),
                                     pad+display_cols*(example_width+pad)))
    curr_ex = 1
    for row_index in range(1, display_rows+1):
        for col_index in range(1, display_cols+1):
            if (curr_ex > num_rows):
                break
            max_val = numpy.amax(numpy.absolute(X[curr_ex-1, :]))
            min_row_idx = pad+(row_index-1)*(example_height+pad)
            max_row_idx = pad+(row_index-1)*(example_height+pad)+example_height
            min_col_idx = pad+(col_index-1)*(example_width+pad)
            max_col_idx = pad+(col_index-1)*(example_width+pad)+example_width
            x_reshape = numpy.reshape(X[curr_ex-1, ], (example_height,
                                                       example_width))
            display_array[min_row_idx:max_row_idx, min_col_idx:max_col_idx] = (
                (1/max_val)*numpy.fliplr(numpy.rot90(x_reshape, 3)))
            curr_ex = curr_ex+1
        if (curr_ex > num_rows):
            break
    pyplot.imshow(display_array, cmap=cm.Greys_r)
    pyplot.axis('off')
    return None


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


def one_vs_all(X, y, num_labels, lamb):
    """ Trains multiple logistic regression classifiers.

    Args:
      X: Matrix of features.
      y: Vector of labels.
      num_labels: Number of classes.
      lamb: Regularization parameter.

    Returns:
      all_theta: Vector of regularized logistic regression parameters (one 
                 per class).

    Raises:
      An error occurs if the number of labels is 0.
    """
    if (num_labels == 0): raise Error('num_labels = 0')
    num_train_ex = X.shape[0]
    num_features = X.shape[1]
    all_theta = numpy.zeros((num_labels, num_features+1))
    ones_vec = numpy.ones((num_train_ex, 1))
    aug_x = numpy.c_[ones_vec, X]
    for label_index in range(0, num_labels):
      theta_vec = numpy.zeros((num_features+1, 1))
      theta_vec_flat = numpy.ndarray.flatten(theta_vec)
      y_arg = (numpy.equal(y, (label_index+1)*numpy.ones((num_train_ex,
                                                          1)))).astype(int)
      fmin_ncg_out = fmin_ncg(compute_cost, theta_vec_flat,
                              fprime=compute_gradient,
                              args=(aug_x, y_arg, num_train_ex, lamb),
                              avextol=1e-10, epsilon=1e-10, maxiter=400,
                              full_output=1)
      theta_opt = numpy.reshape(fmin_ncg_out[0], (1, num_features+1), order='F')
      all_theta[label_index, :] = theta_opt
    return all_theta


def predict_one_vs_all(X, all_theta):
    """ Performs label prediction on training data.

    Args:
      X: Matrix of features.
      all_theta: Vector of regularized logistic regression parameters (one 
                 per class).

    Returns:
      p: Vector of predicted class labels (one per example).

    Raises:
      An error occurs if the number of training examples is 0.
    """
    num_train_ex = X.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    ones_vec = numpy.ones((num_train_ex, 1))
    aug_x = numpy.c_[ones_vec, X]
    sigmoid_array = compute_sigmoid(numpy.dot(aug_x,
                                              numpy.transpose(all_theta)))
    p = numpy.argmax(sigmoid_array, axis=1)
    for example_index in range(0, num_train_ex):
        p[example_index] = p[example_index]+1
    return p


def main():
    """ Main function

    Raises:
      An error occurs if the number of training examples is 0.
    """
    print("Loading and Visualizing Data ...")
    digit_data = numpy.genfromtxt("../digitData.txt", delimiter=",")
    num_train_ex = digit_data.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = digit_data.shape[1]-1
    x_mat = digit_data[:, 0:num_features]
    y_vec = digit_data[:, num_features:(num_features+1)]

    # Randomly select 100 data points to display
    rand_indices = numpy.random.permutation(num_train_ex)
    x_mat_sel = x_mat[rand_indices[0], :]
    for rand_index in range(1, 100):
        x_mat_sel = numpy.vstack([x_mat_sel,
                                  x_mat[rand_indices[rand_index], :]])
    return_code = display_data(x_mat_sel)
    pyplot.show()
    input("Program paused. Press enter to continue.")

    # Train one logistic regression classifier for each digit
    print("\n")
    print("Training One-vs-All Logistic Regression...")
    lamb = 0.1
    num_labels = 10
    all_theta = one_vs_all(x_mat, y_vec, num_labels, lamb)
    input("Program paused. Press enter to continue.")

    # Perform one-versus-all classification using logistic regression
    training_predict = predict_one_vs_all(x_mat, all_theta)
    num_train_match = 0
    for train_index in range(0, num_train_ex):
        if (training_predict[train_index] == y_vec[train_index]):
            num_train_match += 1
    print("\n")
    print("Training Set Accuracy: %.6f" % (100*num_train_match/num_train_ex))

# Call main function
if __name__ == "__main__":
    main()
