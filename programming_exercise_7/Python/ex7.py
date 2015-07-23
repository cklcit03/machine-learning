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
# Programming Exercise 7: K-Means Clustering
# Problem: Apply K-Means Clustering to image compression
from matplotlib import colors
from matplotlib import pyplot
import numpy
import png


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def find_closest_centroids(X, curr_cent):
    """ Finds closest centroids for input data using current centroid
        assignments.

    Args:
      X: Matrix of data features.
      curr_cent: Matrix of current centroid positions.

    Returns:
      centroid_idx: Vector where each entry contains index of closest centroid
                    to corresponding example.

    Raises:
      An error occurs if the number of centroids is 0.
      An error occurs if the number of data examples is 0.
    """
    num_centroids = curr_cent.shape[0]
    if (num_centroids == 0): raise Error('num_centroids == 0')
    num_data = X.shape[0]
    if (num_data == 0): raise Error('num_data == 0')
    centroid_idx = numpy.zeros((num_data, 1))
    for data in range(0, num_data):
        centroid_idx[data] = 0
        min_distance = (
            numpy.sqrt(numpy.sum(numpy.multiply(X[data, :]-curr_cent[0, :],
                                                X[data, :]-curr_cent[0, :]))))
        for centroid in range(1, num_centroids):
            tmp_distance = (
                numpy.sqrt(numpy.sum(numpy.multiply(X[data,
                                                      :]-curr_cent[centroid, :],
                                                    X[data,
                                                      :]-curr_cent[centroid,
                                                                  :]))))
            if (tmp_distance < min_distance):
                min_distance = tmp_distance
                centroid_idx[data] = centroid
    return centroid_idx


def compute_centroids(X, centroid_indices, num_centroids):
    """ Updates centroids for input data using current centroid assignments.

    Args:
      X: Matrix of data features.
      centroid_indices: Vector where each entry contains index of closest 
                        centroid to corresponding example.
      num_centroids: Number of centroids.

    Returns:
      centroid_array: Matrix of centroid positions, where each centroid is the
                      mean of the points assigned to it.

    Raises:
      An error occurs if the number of centroids is 0.
      An error occurs if the number of data features is 0.
    """
    if (num_centroids == 0): raise Error('num_centroids == 0')
    num_features = X.shape[1]
    if (num_features == 0): raise Error('num_features == 0')
    centroid_array = numpy.zeros((num_centroids, num_features))
    for centroid in range(0, num_centroids):
        is_centroid_idx = (centroid_indices == centroid)
        sum_centroid_points = numpy.dot(numpy.transpose(is_centroid_idx), X)
        centroid_array[centroid, :] = (
            sum_centroid_points/numpy.sum(is_centroid_idx))
    return centroid_array


def run_k_means(X, init_centroids, max_iter, plot_flag):
    """ Runs K-Means Clustering on input data.

    Args:
      X: Matrix of data features.
      init_centroids: Matrix of initial centroid positions.
      max_iter: Maximum number of iterations for K-Means Clustering.
      plot_flag: Boolean that indicates whether progress of K-Means Clustering
                 should be plotted.

    Returns:
      return_list: List of two objects.
                   centroid_indices: Vector where each entry contains index of 
                                     closest centroid to corresponding example.
                   curr_centroids: Matrix of centroid positions, where each 
                                   centroid is the mean of the points assigned
                                   to it.

    Raises:
      An error occurs if the maximum number of iterations is 0.
      An error occurs if the number of data examples is 0.
      An error occurs if the number of centroids is 0.
    """
    if (max_iter == 0): raise Error('max_iter == 0')
    num_data = X.shape[0]
    if (num_data == 0): raise Error('num_data == 0')
    num_centroids = init_centroids.shape[0]
    if (num_centroids == 0): raise Error('num_centroids == 0')
    centroid_idx = numpy.zeros((num_data, 1))
    curr_cent = init_centroids

    # Create an array that stores all centroids
    # This array will be useful for plotting
    all_centroids = numpy.zeros((num_centroids*max_iter,
                                 init_centroids.shape[1]))
    for centroid_idx in range(0, num_centroids):
        all_centroids[centroid_idx, :] = init_centroids[centroid_idx, :]
    for iter_index in range(0, max_iter):
        print("K-Means iteration %d/%d..." % (iter_index+1, max_iter))

        # Assign centroid to each datum
        centroid_indices = find_closest_centroids(X, curr_cent)

        # Plot progress of algorithm
        if (plot_flag == True):
            plot_progress_k_means(X, all_centroids, centroid_indices,
                                  num_centroids, iter_index)
            pyplot.show()
            prev_centroids = curr_cent
            input("Program paused. Press enter to continue.")
            print("")

        # Compute updated centroids
        curr_cent = compute_centroids(X, centroid_indices, num_centroids)
        if (iter_index < (max_iter-1)):
            for centroid_idx in range(0, num_centroids):
                all_centroids[(iter_index+1)*num_centroids+centroid_idx, :] = (
                    curr_cent[centroid_idx, :])
    return_list = {'centroid_indices': centroid_indices, 'curr_cent': curr_cent}
    return return_list


def plot_progress_k_means(X, all_centroids, centroid_indices, num_centroids,
                          iter_index):
    """ Displays progress of K-Means Clustering.

    Args:
      X: Matrix of data features.
      all_centroids: Matrix of all (current and previous) centroid positions.
      centroid_indices: Vector where each entry contains index of closest 
                        centroid to corresponding example.
      num_centroids: Number of centroids.
      iter_index: Current iteration of K-Means Clustering.

    Returns:
      None.

    Raises:
      An error occurs if the number of centroids is 0.
    """
    if (num_centroids == 0): raise Error('num_centroids == 0')

    # Plot input data
    return_code = plot_data_points(X, centroid_indices, num_centroids)

    # Plot centroids as black X's
    centroids = pyplot.scatter(all_centroids[0:(iter_index+1)*num_centroids, 0],
                               all_centroids[0:(iter_index+1)*num_centroids, 1],
                               s=80, marker='x', color='k')

    # Plot history of centroids with lines
    for iter2_index in range(0, iter_index):
        for centroid in range(0, num_centroids):
            return_code = (
                draw_line(all_centroids[(iter2_index+1)*num_centroids+centroid,
                                        :],
                          all_centroids[iter2_index*num_centroids+centroid, :]))
    return None


def plot_data_points(X, centroid_indices, num_centroids):
    """ Plots input data with colors according to current cluster assignments.

    Args:
      X: Matrix of data features.
      centroid_indices: Vector where each entry contains index of closest 
                        centroid to corresponding example.
      num_centroids: Number of centroids.

    Returns:
      None.

    Raises:
      An error occurs if the number of data examples is 0.
    """
    num_data = X.shape[0]
    if (num_data == 0): raise Error('num_data == 0')
    palette = numpy.zeros((num_centroids+1, 3))
    for centroid_idx in range(0, num_centroids+1):
        hsv_h = centroid_idx/(num_centroids+1)
        hsv_s = 1
        hsv_v = 1
        palette[centroid_idx, :] = colors.hsv_to_rgb(numpy.r_[hsv_h, hsv_s,
                                                              hsv_v])
    curr_colors = numpy.zeros((num_data, 3))
    for data_idx in range(0, num_data):
        curr_centroid_idx = centroid_indices[data_idx].astype(int)
        curr_colors[curr_centroid_idx, 0] = palette[curr_centroid_idx, 0]
        curr_colors[curr_centroid_idx, 1] = palette[curr_centroid_idx, 1]
        curr_colors[curr_centroid_idx, 2] = palette[curr_centroid_idx, 2]
        pyplot.scatter(X[data_idx, 0], X[data_idx, 1], s=80, marker='o',
                       facecolors='none',
                       edgecolors=curr_colors[curr_centroid_idx, :])
    return None


def draw_line(start_point,end_point):
    """ Draws line between input points.

    Args:
      start_point: 2-D vector that represents starting point.
      end_point: 2-D vector that represents ending point.

    Returns:
      None.
    """
    pyplot.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                'b')
    return None


def k_means_init_centroids(X, num_centroids):
    """ Initializes centroids for input data.

    Args:
      X: Matrix of data features.
      num_centroids: Number of centroids.

    Returns:
      init_centroids: Matrix of initial centroid positions.

    Raises:
      An error occurs if the number of centroids is 0.
    """
    if (num_centroids == 0): raise Error('num_centroids == 0')
    rand_indices = numpy.random.permutation(X.shape[0])
    init_centroids = numpy.zeros((num_centroids, 3))
    for centroid_idx in range(0, num_centroids):
        init_centroids[centroid_idx, :] = X[rand_indices[centroid_idx, ], :]
    return init_centroids


def main():
    """ Main function
    """
    print("Finding closest centroids.")
    exercise_7_data_2 = numpy.genfromtxt("../ex7data2.txt", delimiter=",")
    num_features = exercise_7_data_2.shape[1]
    x_mat = exercise_7_data_2[:, 0:num_features]

    # Select an initial set of centroids
    num_centroids = 3
    initial_centroids = numpy.r_[numpy.c_[3, 3], numpy.c_[6, 2], numpy.c_[8, 5]]

    # Find closest centroids for example data using initial centroids
    centroid_indices = find_closest_centroids(x_mat, initial_centroids)
    print("Closest centroids for the first 3 examples:")
    print("%s" % numpy.array_str(numpy.transpose(centroid_indices[0:3])+1))
    print("(the closest centroids should be 1, 3, 2 respectively)")
    input("Program paused. Press enter to continue.")
    print("")

    # Update centroids for example data
    print("Computing centroids means.")
    updated_centroids = compute_centroids(x_mat, centroid_indices,
                                          num_centroids)
    print("Centroids computed after initial finding of closest centroids:")
    print("%s" % numpy.array_str(numpy.round(updated_centroids[0, :], 6)))
    print("%s" % numpy.array_str(numpy.round(updated_centroids[1, :], 6)))
    print("%s" % numpy.array_str(numpy.round(updated_centroids[2, :], 6)))
    print("(the centroids should be")
    print("   [ 2.428301 3.157924 ]")
    print("   [ 5.813503 2.633656 ]")
    print("   [ 7.119387 3.616684 ]")
    input("Program paused. Press enter to continue.")
    print("")

    # Run K-Means Clustering on an example dataset
    print("Running K-Means clustering on example dataset.")
    max_iter = 10
    k_means_list = run_k_means(x_mat, initial_centroids, max_iter, True)
    print("K-Means Done.")
    input("Program paused. Press enter to continue.")
    print("")

    # Use K-Means Clustering to compress an image
    print("Running K-Means clustering on pixels from an image.")
    bird_small_file = open('../bird_small.png', 'rb')
    bird_small_reader = png.Reader(file=bird_small_file)
    row_count, col_count, bird_small, meta = bird_small_reader.asDirect()
    plane_count = meta['planes']
    bird_small_2d = numpy.zeros((row_count, col_count*plane_count))
    for row_index, one_boxed_row_flat_pixels in enumerate(bird_small):
        bird_small_2d[row_index, :] = one_boxed_row_flat_pixels
        bird_small_2d[row_index, :] = (1/255)*bird_small_2d[row_index,
                                                            :].astype(float)
    bird_small_reshape = bird_small_2d.reshape((row_count*col_count, 3))
    bird_small_3d_reshape = bird_small_reshape.reshape((row_count, col_count,
                                                        3))
    num_centroids = 16
    max_iter = 10

    # Initialize centroids randomly
    initial_centroids = k_means_init_centroids(bird_small_reshape,
                                               num_centroids)
    k_means_list = run_k_means(bird_small_reshape, initial_centroids, max_iter,
                               False)
    input("Program paused. Press enter to continue.")
    print("")

    # Use the output clusters to compress this image
    print("Applying K-Means to compress an image.")
    curr_cent = k_means_list['curr_cent']
    centroid_indices = find_closest_centroids(bird_small_reshape, curr_cent)
    bird_small_recovered = numpy.zeros((row_count*col_count, 3))
    for row_index in range(0, bird_small_recovered.shape[0]):
        curr_index = centroid_indices[row_index, :].astype(int)
        bird_small_recovered[row_index, :] = curr_cent[curr_index, :]
    bird_small_recovered = bird_small_recovered.reshape((row_count, col_count,
                                                         3))

    # Display original and compressed images side-by-side
    fig, (ax1, ax2) = pyplot.subplots(1, 2)
    ax1.set_title('Original')
    im1 = ax1.imshow(bird_small_3d_reshape)
    recovered_title = 'Compressed, with %d colors.' % num_centroids
    ax2.set_title(recovered_title)
    im2 = ax2.imshow(bird_small_recovered)
    pyplot.show()

# Call main function
if __name__ == "__main__":
    main()
