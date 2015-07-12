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
# Problem: Predict chances of university admission for an applicant given data
# for admissions decisions and test scores of various applicants
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
    pyplot.hold(False)
    pyplot.ylabel('Exam 2 score', fontsize=18)
    pyplot.xlabel('Exam 1 score', fontsize=18)
    return None


def plot_decision_boundary(x, y, theta):
    """ Plots decision boundary.

    Args:
      x: Features that have already been plotted.
      y: Data labels.
      theta: Parameter that determines slope of decision boundary.

    Returns:
      None.
    """
    plot_data(numpy.c_[x[:, 1], x[:, 2]], y)
    pyplot.hold(True)
    y_line_vals = (theta[0]+theta[1]*x[:, 1])/(-1*theta[2])
    pyplot.plot(x[:, 1], y_line_vals, 'b-', markersize=18)
    pyplot.hold(False)
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


def compute_cost(theta, X, y, num_train_ex):
    """ Computes cost function J(\theta).

    Args:
      theta: Vector of parameters for logistic regression.
      X: Matrix of features.
      y: Vector of labels.
      num_train_ex: Number of training examples.

    Returns:
      j_theta: Logistic regression cost.

    Raises:
      An error occurs if the number of features is 0.
      An error occurs if the number of training examples is 0.
    """
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = X.shape[1]
    if num_features == 0: raise Error('num_features = 0')
    theta = numpy.reshape(theta, (num_features, 1), order='F')
    h_theta = compute_sigmoid(numpy.dot(X, theta))
    j_theta = (numpy.sum(numpy.subtract(numpy.multiply(-y, numpy.log(h_theta)),
                                        numpy.multiply((1-y),
                                                       numpy.log(1-h_theta))),
                         axis=0))/num_train_ex
    return j_theta


def compute_gradient(theta, X, y, num_train_ex):
    """ Computes gradient of cost function J(\theta).

    Args:
      theta: Vector of parameters for logistic regression.
      X: Matrix of features.
      y: Vector of labels.
      num_train_ex: Number of training examples.

    Returns:
      grad_array_flat: Vector of logistic regression gradients
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
    for grad_index in range(0, num_features):
        grad_term = numpy.multiply(numpy.reshape(X[:, grad_index],
                                                 (num_train_ex, 1)),
                                   numpy.subtract(h_theta, y))
        grad_array[grad_index] = (numpy.sum(grad_term, axis=0))/num_train_ex
    grad_array_flat = numpy.ndarray.flatten(grad_array)
    return grad_array_flat


def compute_cost_grad_list(theta, X, y):
    """ Aggregates computed cost and gradient.

    Args:
      theta: Vector of parameters for logistic regression.
      X: Matrix of features.
      y: Vector of labels.


    Returns:
      return_list: List of two objects.
                   j_theta: Updated vector of parameters for logistic
                            regression.
                   grad_array: Updated vector of logistic regression gradients 
                               (one per feature).
    """
    num_features = X.shape[1]
    num_train_ex = y.shape[0]
    j_theta = compute_cost(theta, X, y, num_train_ex)
    grad_array_flat = compute_gradient(theta, X, y, num_train_ex)
    grad_array = numpy.reshape(grad_array_flat, (num_features, 1), order='F')
    return_list = {'j_theta': j_theta, 'grad_array': grad_array}
    return return_list


def label_prediction(X, theta):
    """ Performs label prediction on training data.

    Args:
      X: Matrix of features.
      theta: Vector of parameters for logistic regression.

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
    print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")
    print("")
    applicant_data = numpy.genfromtxt("../applicantData.txt", delimiter=",")
    return_code = plot_data(numpy.c_[applicant_data[:, 0],
                                     applicant_data[:, 1]],
                            applicant_data[:, 2])
    pyplot.legend(('Admitted', 'Not admitted'), loc='lower right')
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")
    num_train_ex = applicant_data.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = applicant_data.shape[1]-1
    ones_vec = numpy.ones((num_train_ex, 1))
    x_mat = numpy.c_[applicant_data[:, 0], applicant_data[:, 1]]
    x_mat_aug = numpy.c_[ones_vec, x_mat]
    y_vec = numpy.reshape(applicant_data[:, 2], (num_train_ex, 1))
    theta_vec = numpy.zeros((num_features+1, 1))

    # Compute initial cost and gradient
    init_compute_cost_list = compute_cost_grad_list(theta_vec, x_mat_aug, y_vec)
    print("Cost at initial theta (zeros): %.6f" %
          init_compute_cost_list['j_theta'])
    print("Gradient at initial theta (zeros):")
    print("%s\n" %
          numpy.array_str(numpy.round(init_compute_cost_list['grad_array'], 6)))
    input("Program paused. Press enter to continue.")

    # Use fmin_ncg to solve for optimum theta and cost
    theta_vec_flat = numpy.ndarray.flatten(theta_vec)
    f_min_ncg_out = fmin_ncg(compute_cost, theta_vec_flat,
                             fprime=compute_gradient, args=(x_mat_aug, y_vec,
                                                            num_train_ex),
                             avextol=1e-10, epsilon=1e-10, maxiter=400,
                             full_output=1)
    theta_opt = numpy.reshape(f_min_ncg_out[0], (num_features+1, 1), order='F')
    print("Cost at theta found by fmin_ncg: %.6f" % f_min_ncg_out[1])
    print("theta:")
    print("%s\n" % numpy.array_str(numpy.round(theta_opt, 6)))
    return_code = plot_decision_boundary(x_mat_aug, y_vec, theta_opt)
    pyplot.legend(('Decision Boundary', 'Admitted', 'Not admitted'),
                  loc='lower left')
    pyplot.show()
    input("Program paused. Press enter to continue.")

    # Predict admission probability for a student with score 45 on exam 1 and
    # score 85 on exam 2
    admission_prob = compute_sigmoid(numpy.dot(numpy.array([1, 45, 85]),
                                               theta_opt))
    print("For a student with scores 45 and 85, we predict an admission probability of %.6f" %
          admission_prob)

    # Compute accuracy on training set
    training_predict = label_prediction(x_mat_aug, theta_opt)
    num_train_match = 0
    for train_index in range(0, num_train_ex):
        if (training_predict[train_index] == y_vec[train_index]):
            num_train_match += 1
    print("Train Accuracy: %.6f" % (100*num_train_match/num_train_ex))
    input("Program paused. Press enter to continue.")

# Call main function
if __name__ == "__main__":
    main()
