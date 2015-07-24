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
# Programming Exercise 8: Anomaly Detection
# Problem: Apply anomaly detection to detect anomalous behavior in servers
from matplotlib import pyplot
import numpy


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def estimate_gaussian(X):
    """ Estimates mean and variance of input (Gaussian) data.

    Args:
      X: Matrix of features.

    Returns:
      return_list: List of two objects.
                   mu_vec: Vector of mean values of features.
                   var_vec: Vector of variances of features.

    Raises:
      An error occurs if the number of features is 0.
    """
    num_feats = X.shape[1]
    if (num_feats == 0): raise Error('num_feats == 0')
    mu_vec = numpy.mean(X, axis=0)
    var_vec = numpy.var(X, axis=0, ddof=1)
    mu_vec = numpy.reshape(mu_vec, (1, num_feats), order='F')
    var_vec = numpy.reshape(var_vec, (1, num_feats), order='F')
    return_list = {'mu_vec': mu_vec, 'var_vec': var_vec}
    return return_list


def multivariate_gaussian(X, mu_vec, var_vec):
    """ Computes multivariate Gaussian PDF for input data.

    Args:
      X: Matrix of features.
      mu_vec: Vector of mean values of features.
      var_vec: Vector of variances of features.

    Returns:
      prob_vec: Vector where each entry contains probability of corresponding 
                example.

    Raises:
      An error occurs if the number of features is 0.
      An error occurs if the number of data examples is 0.
    """
    num_feats = X.shape[1]
    if (num_feats == 0): raise Error('num_feats == 0')
    num_data = X.shape[0]
    if (num_data == 0): raise Error('num_data == 0')
    var_mat = numpy.diag(numpy.ravel(var_vec))
    prob_vec = numpy.zeros((num_data, 1))
    for data_index in range(0, num_data):
        num_term = (
            (numpy.exp(-0.5*numpy.dot(numpy.dot(X[data_index, ]-mu_vec,
                                                numpy.linalg.inv(var_mat)),
                                      numpy.transpose(X[data_index,
                                                        ]-mu_vec)))))
        den_term = (
            (numpy.power(2*numpy.pi,
                         0.5*num_feats)*numpy.sqrt(numpy.linalg.det(var_mat))))
        prob_vec[data_index] = num_term/den_term
    return prob_vec


def visualize_fit(X, mu_vec, var_vec):
    """ Plots dataset and estimated Gaussian distribution.

    Args:
      X: Matrix of features.
      mu_vec: Vector of mean values of features.
      var_vec: Vector of variances of features.

    Returns:
      None.
    """
    pyplot.scatter(X[:, 0], X[:, 1], s=80, marker='x', color='b')
    pyplot.ylabel('Throughput (mb/s)', fontsize=18)
    pyplot.xlabel('Latency (ms)', fontsize=18)
    pyplot.hold(True)
    u_vals = numpy.linspace(0, 35, num=71)
    v_vals = numpy.linspace(0, 35, num=71)
    z_vals = numpy.zeros((u_vals.shape[0], v_vals.shape[0]))
    for u_index in range(0, u_vals.shape[0]):
        for v_index in range(0, v_vals.shape[0]):
            z_vals[u_index, v_index] = (
                multivariate_gaussian(numpy.c_[u_vals[u_index],
                                               v_vals[v_index]], mu_vec,
                                      var_vec))
    z_vals_trans = numpy.transpose(z_vals)
    u_vals_x, v_vals_y = numpy.meshgrid(u_vals, v_vals)
    z_vals_reshape = z_vals_trans.reshape(u_vals_x.shape)
    exp_seq = numpy.linspace(-20, 1, num=8)
    pow_exp_seq = numpy.power(10, exp_seq)
    pyplot.contour(u_vals, v_vals, z_vals_reshape, pow_exp_seq)
    pyplot.hold(False)
    return None


def select_threshold(y, p):
    """ Finds the best threshold for detecting anomalies.

    Args:
      y: Cross-validation labels.
      p: Vector where each entry contains probability of corresponding
         cross-validation example.

    Returns:
      return_list: List of two objects.
                   best_f1: F1 score.
                   best_epsilon: Selected threshold.
    """
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    step_size = 0.001*(numpy.max(p)-numpy.min(p))
    epsilon_seq = (
        numpy.linspace(numpy.min(p), numpy.max(p),
                       num=numpy.ceil((numpy.max(p)-numpy.min(p))/step_size)))
    for epsilon_index in range(0, 1000):
        curr_epsilon = epsilon_seq[epsilon_index, ]
        predictions = (p < curr_epsilon)
        num_true_positives = numpy.sum((predictions == 1) & (y == 1))
        if (num_true_positives > 0):
            num_false_positives = numpy.sum((predictions == 1) & (y == 0))
            num_false_negatives = numpy.sum((predictions == 0) & (y == 1))
            precision_val = (
                num_true_positives/(num_true_positives+num_false_positives))
            recall_val = (
                num_true_positives/(num_true_positives+num_false_negatives))
            f1 = (2*precision_val*recall_val)/(precision_val+recall_val)
            if (f1 > best_f1):
                best_f1 = f1
                best_epsilon = curr_epsilon
    return_list = {'best_f1': best_f1, 'best_epsilon': best_epsilon}
    return return_list


def main():
    """ Main function
    """
    print("Visualizing example dataset for outlier detection.")
    server_data_1 = numpy.genfromtxt("../serverData1.txt", delimiter=",")
    num_feats = server_data_1.shape[1]
    x_mat = server_data_1[:, 0:num_feats]
    pyplot.scatter(x_mat[:, 0], x_mat[:, 1], s=80, marker='x', color='b')
    pyplot.ylabel('Throughput (mb/s)', fontsize=18)
    pyplot.xlabel('Latency (ms)', fontsize=18)
    pyplot.axis([0, 30, 0, 30])
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Estimate (Gaussian) statistics of this dataset
    print("Visualizing Gaussian fit.")
    estimate_gaussian_list = estimate_gaussian(x_mat)
    mu_vec = estimate_gaussian_list['mu_vec']
    var_vec = estimate_gaussian_list['var_vec']
    prob_vec = multivariate_gaussian(x_mat, mu_vec, var_vec)
    return_code = visualize_fit(x_mat, mu_vec, var_vec)
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Use a cross-validation set to find outliers
    server_val_data_1 = numpy.genfromtxt("../serverValData1.txt", delimiter=",")
    num_val_feats = server_val_data_1.shape[1]-1
    x_val_mat = server_val_data_1[:, 0:num_val_feats]
    y_val_vec = server_val_data_1[:, num_val_feats:(num_val_feats+1)]
    prob_val_vec = multivariate_gaussian(x_val_mat, mu_vec, var_vec)
    select_threshold_list = select_threshold(y_val_vec, prob_val_vec)
    best_epsilon = select_threshold_list['best_epsilon']
    best_f1 = select_threshold_list['best_f1']
    print("Best epsilon found using cross-validation: %e" % best_epsilon)
    print("Best F1 on Cross Validation Set:  %f" % best_f1)
    print("   (you should see a value epsilon of about 8.99e-05)")
    outlier_indices = (numpy.array(numpy.where(prob_vec < best_epsilon)))[0, :]
    return_code = visualize_fit(x_mat, mu_vec, var_vec)
    pyplot.hold(True)
    outliers = pyplot.scatter(x_mat[outlier_indices, 0],
                              x_mat[outlier_indices, 1], s=80, marker='o',
                              facecolors='none', edgecolors='r')
    pyplot.hold(False)
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Detect anomalies in another dataset
    server_data_2 = numpy.genfromtxt("../serverData2.txt", delimiter=",")
    num_feats = server_data_2.shape[1]
    x_mat = server_data_2[:, 0:num_feats]

    # Estimate (Gaussian) statistics of this dataset
    estimate_gaussian_list = estimate_gaussian(x_mat)
    mu_vec = estimate_gaussian_list['mu_vec']
    var_vec = estimate_gaussian_list['var_vec']
    prob_vec = multivariate_gaussian(x_mat, mu_vec, var_vec)

    # Use a cross-validation set to find outliers in this dataset
    server_val_data_2 = numpy.genfromtxt("../serverValData2.txt", delimiter=",")
    num_val_feats = server_val_data_2.shape[1]-1
    x_val_mat = server_val_data_2[:, 0:num_val_feats]
    y_val_vec = server_val_data_2[:, num_val_feats:(num_val_feats+1)]
    prob_val_vec = multivariate_gaussian(x_val_mat, mu_vec, var_vec)
    select_threshold_list = select_threshold(y_val_vec, prob_val_vec)
    best_epsilon = select_threshold_list['best_epsilon']
    best_f1 = select_threshold_list['best_f1']
    outlier_indices = (numpy.array(numpy.where(prob_vec < best_epsilon)))[0, :]
    print("Best epsilon found using cross-validation: %e" % best_epsilon)
    print("Best F1 on Cross Validation Set:  %f" % best_f1)
    print("# Outliers found: %d" % outlier_indices.size)
    print("   (you should see a value epsilon of about 1.38e-18)")
    input("Program paused. Press enter to continue.")
    print("")

# Call main function
if __name__ == "__main__":
    main()
