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
# Use parameters trained by a neural network for prediction
from matplotlib import cm
from matplotlib import pyplot
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


def predict(theta_1, theta_2, X):
    """ Performs label prediction on training data.

    Args:
      theta_1: Matrix of neural network parameters (map from input layer to
               hidden layer).
      theta_2: Matrix of neural network parameters (map from hidden layer to
               output layer).
      X: Matrix of features.

    Returns:
      p: Vector of predicted class labels (one per example).

    Raises:
      An error occurs if the number of training examples is 0.
    """
    num_train_ex = X.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    ones_vec = numpy.ones((num_train_ex, 1))
    aug_x = numpy.c_[ones_vec, X]
    hidden_layer_activation = (
        compute_sigmoid(numpy.dot(aug_x, numpy.transpose(theta_1))))
    ones_vec_mod = numpy.ones((hidden_layer_activation.shape[0], 1))
    hidden_layer_activation_mod = numpy.c_[ones_vec_mod,
                                           hidden_layer_activation]
    output_layer_activation = (
        compute_sigmoid(numpy.dot(hidden_layer_activation_mod,
                                  numpy.transpose(theta_2))))
    p = numpy.argmax(output_layer_activation, axis=1)
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

    # Load two files that contain parameters trained by a neural network into R
    print("\n")
    print("Loading Saved Neural Network Parameters ...")
    theta_1_mat = numpy.genfromtxt("../Theta1.txt", delimiter=",")
    theta_2_mat = numpy.genfromtxt("../Theta2.txt", delimiter=",")

    # Perform one-versus-all classification using trained parameters
    training_predict = predict(theta_1_mat, theta_2_mat, x_mat)
    num_train_match = 0
    for train_index in range(0, num_train_ex):
        if (training_predict[train_index] == y_vec[train_index]):
            num_train_match += 1
    print("\n")
    print("Training Set Accuracy: %.6f" % (100*num_train_match/num_train_ex))
    input("Program paused. Press enter to continue.")

    # Display example images along with predictions from neural network
    rand_indices = numpy.random.permutation(num_train_ex)
    for example_index in range(0, 10):
        print("\n")
        print("Displaying Example Image")
        print("\n")
        x_mat_sel = numpy.reshape(x_mat[rand_indices[example_index], :],
                                  (1, num_features), order='F')
        return_code = display_data(x_mat_sel)
        pyplot.show()
        example_predict = predict(theta_1_mat, theta_2_mat, x_mat_sel)
        print("Neural Network Prediction: %d (digit %d)" % (example_predict,
                                                            example_predict%10))
        input("Program paused. Press enter to continue.")

# Call main function
if __name__ == "__main__":
    main()
