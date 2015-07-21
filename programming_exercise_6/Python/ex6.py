# Copyright (C) 2015  Caleb Lo
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
# Programming Exercise 6: Support Vector Machines
# Problem: Use SVMs to learn decision boundaries for various example datasets
from matplotlib import pyplot
from sklearn import svm
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
    return None


def plot_linear_decision_boundary(x, y, theta):
    """ Plots linear decision boundary.

    Args:
      x: Features that have already been plotted.
      y: Data labels.
      theta: Parameter that determines slope of decision boundary.

    Returns:
      None.
    """
    plot_data(numpy.c_[x[:, 0], x[:, 1]], y)
    pyplot.hold(True)
    y_line_vals = (theta[0]+theta[1]*x[:, 0])/(-1*theta[2])
    pyplot.plot(x[:, 0], y_line_vals, 'b-', markersize=18)
    pyplot.hold(False)
    return None


def plot_decision_boundary(x, y, svm_model):
    """ Plots non-linear decision boundary learned via SVM.

    Args:
      x: Features to be plotted.
      y: Data labels.
      svm_model: Fitted SVM model.

    Returns:
      None.
    """
    plot_data(numpy.c_[x[:, 0], x[:, 1]], y)
    pyplot.hold(True)
    x1 = numpy.linspace(numpy.amin(x[:, 0], axis=0),
                        numpy.amax(x[:, 0], axis=0), num=100)
    x2 = numpy.linspace(numpy.amin(x[:, 1], axis=0),
                        numpy.amax(x[:, 1], axis=0), num=100)
    j_vals = numpy.zeros((x1.shape[0], x2.shape[0]))
    for x1_index in range(0, x1.shape[0]):
        for x2_index in range(0, x2.shape[0]):
            j_vals[x1_index, x2_index] = svm_model.predict([[x1[x1_index, ],
                                                             x2[x2_index, ]]])
    j_vals_trans = numpy.transpose(j_vals)
    x1_x, x2_y = numpy.meshgrid(x1, x2)
    j_vals_reshape = j_vals_trans.reshape(x1_x.shape)
    pyplot.contour(x1, x2, j_vals_reshape, 1)
    pyplot.hold(False)
    return None


def dataset3_params(X1, y1, x_val, y_val):
    """ Selects optimal learning parameters for radial basis SVM.

    Args:
      X1: Matrix of training features.
      y1: Vector of training labels.
      x_val: Matrix of cross-validation features.
      y_val: Vector of cross-validation labels.

    Returns:
      return_list: List of two objects.
                   C: Best (in terms of minimum prediction error for 
                      cross-validation data) value that controls penalty for 
                      misclassified training examples.
                   sigma: Best (in terms of minimum prediction error for 
                          cross-validation data) value that determines how fast
                          similarity metric decreases as examples are further 
                          apart.

    Raises:
      An error occurs if the number of cross-validation examples is 0.
    """
    num_val_ex = y_val.shape[0]
    if (num_val_ex == 0): raise Error('num_val_ex == 0')
    C = 1
    sigma = 0.3
    c_arr = numpy.c_[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    num_c = c_arr.shape[1]
    c_arr = numpy.reshape(c_arr, (num_c, 1), order='F')
    sig_arr = numpy.c_[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    num_sigma = sig_arr.shape[1]
    sig_arr = numpy.reshape(sig_arr, (num_sigma, 1), order='F')
    best_pred_err = 1000000
    for c_index in range(0, num_c):
        for sigma_index in range(0, num_sigma):
            svm_model_tmp = svm.SVC(C=c_arr[c_index], kernel='rbf',
                                    gamma=1/(2*numpy.power(sig_arr[sigma_index],
                                                           2)))
            svm_model_tmp.fit(X1, y1)
            pred_vec = svm_model_tmp.predict(x_val)
            curr_pred_err = 0
            for val_index in range(0, num_val_ex):
                if (pred_vec[val_index] != y_val[val_index]):
                    curr_pred_err = curr_pred_err + 1
            if (curr_pred_err < best_pred_err):
                c_best = c_arr[c_index]
                sigma_best = sig_arr[sigma_index]
                best_pred_err = curr_pred_err
    C = c_best
    sigma = sigma_best
    return_list = {'C': C, 'sigma': sigma}
    return return_list


def main():
    """ Main function
    """
    print("Loading and Visualizing Data ...")
    example_data_1 = numpy.genfromtxt("../exampleData1.txt", delimiter=",")
    num_train_ex = example_data_1.shape[0]
    num_features = example_data_1.shape[1]-1

    # Plot data
    x_mat = example_data_1[:, 0:num_features]
    y_vec = example_data_1[:, num_features]
    return_code = plot_data(x_mat, y_vec)
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Train linear SVM on data
    print("Training Linear SVM ...")
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(x_mat, y_vec)
    theta_opt = numpy.c_[svm_model.intercept_, svm_model.coef_]
    return_code = plot_linear_decision_boundary(x_mat, y_vec,
                                                numpy.transpose(theta_opt))
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Load another dataset and plot it
    example_data_2 = numpy.genfromtxt("../exampleData2.txt", delimiter=",")
    num_train_ex_2 = example_data_2.shape[0]
    num_features_2 = example_data_2.shape[1]-1
    x_mat_2 = example_data_2[:, 0:num_features_2]
    y_vec_2 = example_data_2[:, num_features_2]
    return_code = plot_data(x_mat_2, y_vec_2)
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Train radial basis SVM on data
    print("Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...")
    sigma_val = 0.1
    svm_model_2 = svm.SVC(kernel='rbf', gamma=1/(2*numpy.power(sigma_val, 2)))
    svm_model_2.fit(x_mat_2, y_vec_2)
    return_code = plot_decision_boundary(x_mat_2, y_vec_2, svm_model_2)
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Load another dataset (along with cross-validation data) and plot it
    example_data_3 = numpy.genfromtxt("../exampleData3.txt", delimiter=",")
    num_train_ex_3 = example_data_3.shape[0]
    num_features_3 = example_data_3.shape[1]-1
    x_mat_3 = example_data_3[:, 0:num_features_3]
    y_vec_3 = example_data_3[:, num_features_3]
    example_val_data_3 = numpy.genfromtxt("../exampleValData3.txt",
                                          delimiter=",")
    num_val_ex_3 = example_val_data_3.shape[0]
    x_val_mat_3 = example_val_data_3[:, 0:num_features_3]
    y_val_vec_3 = example_val_data_3[:, num_features_3]
    return_code = plot_data(x_mat_3, y_vec_3)
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Use cross-validation data to train radial basis SVM
    dataset3_params_list = dataset3_params(x_mat_3, y_vec_3, x_val_mat_3,
                                           y_val_vec_3)
    svm_model_3 = svm.SVC(C=dataset3_params_list['C'], kernel='rbf',
                          gamma=1/(2*numpy.power(dataset3_params_list['sigma'],
                                                 2)))
    svm_model_3.fit(x_mat_3, y_vec_3)
    return_code = plot_decision_boundary(x_mat_3, y_vec_3, svm_model_3)
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

# Call main function
if __name__ == "__main__":
    main()
