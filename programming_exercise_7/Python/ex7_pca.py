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
# Programming Exercise 7: Principal Component Analysis (PCA)
# Problem: Use PCA for dimensionality reduction
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy
import png


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


def pca(X):
    """ Runs PCA on input data.

    Args:
      X: Matrix of features.

    Returns:
      return_list: List of two objects.
                   left_sing_vec: Matrix whose columns contain the left singular
                                  vectors of X.
                   sing_val: Vector containing the singular values of X.

    Raises:
      An error occurs if the number of training examples is 0.
    """
    num_train_ex = X.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    cov_mat = (1/num_train_ex)*numpy.dot(X.conj().T, X)
    U, s, V = numpy.linalg.svd(cov_mat)
    return_list = {"left_sing_vec": U, "sing_val": s}
    return return_list


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


def project_data(X, sing_vec, num_dim):
    """ Projects input data onto reduced-dimensional space.

    Args:
      X: Matrix of features.
      sing_vec: Matrix whose columns contain the left singular vectors of X.
      num_dim: Number of dimensions for dimensionality reduction.

    Returns:
      mapped_data: Matrix of projected data, where each example has been
                   projected onto the top num_dim components of sing_vec.

    Raises:
      An error occurs if the number of dimensions is 0.
    """
    if (num_dim == 0): raise Error('num_dim = 0')
    reduced_sing_vec = sing_vec[:, 0:num_dim]
    mapped_data = numpy.dot(X, reduced_sing_vec)
    return mapped_data


def recover_data(X, sing_vec, num_dim):
    """ Projects input data onto original space.

    Args:
      X: Matrix of projected features.
      sing_vec: Matrix whose columns contain the left singular vectors of
                original features.
      num_dim: Number of dimensions for projected features.

    Returns:
      recovered_data: Matrix of recovered data, where each projected example has
                      been projected onto the original high-dimensional space.

    Raises:
      An error occurs if the number of dimensions is 0.
    """
    if (num_dim == 0): raise Error('num_dim = 0')
    reduced_sing_vec = sing_vec[:, 0:num_dim]
    recovered_data = numpy.dot(X, reduced_sing_vec.T.conj())
    return recovered_data


def display_data(X, axes=None):
    """ Displays 2D data in a grid.

    Args:
      X: Matrix of 2D data that will be displayed using imshow().
      axes: Flag that determines whether a subplot is being used.

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

    # If axes is not None, then we are using a subplot
    if (axes == None):
        pyplot.imshow(display_array, cmap=cm.Greys_r)
        pyplot.axis('off')
    else:
        axes.imshow(display_array, cmap=cm.Greys_r)
        axes.axis('off')
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


def main():
    """ Main function

    Raises:
      An error occurs if the number of training examples is 0.
    """
    print("Visualizing example dataset for PCA.")
    exercise_7_data_1 = numpy.genfromtxt("../ex7data1.txt", delimiter=",")
    num_train_ex = exercise_7_data_1.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = exercise_7_data_1.shape[1]
    x_mat = exercise_7_data_1[:, 0:num_features]

    # Visualize input data
    pyplot.scatter(x_mat[:, 0], x_mat[:, 1], s=80, facecolors='none',
                   edgecolors='b')
    axes = pyplot.gca()
    axes.set_xlim([0.5, 6.5])
    axes.set_ylim([2, 8])
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Run PCA on input data
    print("Running PCA on example dataset.")
    feat_norm_list = feature_normalize(x_mat)
    pca_list = pca(feat_norm_list['x_norm'])

    # Draw eigenvectors centered at mean of data
    pyplot.scatter(x_mat[:, 0], x_mat[:, 1], s=80, facecolors='none',
                   edgecolors='b')
    axes = pyplot.gca()
    axes.set_xlim([0.5, 6.5])
    axes.set_ylim([2, 8])
    pyplot.hold(True)
    return_code = draw_line(feat_norm_list['mu_vec'], feat_norm_list['mu_vec']+(
        1.5*pca_list['sing_val'][0]*pca_list['left_sing_vec'][:, 0]))
    pyplot.hold(True)
    return_code = draw_line(feat_norm_list['mu_vec'], feat_norm_list['mu_vec']+(
        1.5*pca_list['sing_val'][1]*pca_list['left_sing_vec'][:, 1]))
    pyplot.show()
    print("Top eigenvector: ")
    print("U(:,1) = %f %f" % (pca_list['left_sing_vec'][0, 0],
                              pca_list['left_sing_vec'][1, 0]))
    print("(you should expect to see -0.707107 -0.707107)")
    input("Program paused. Press enter to continue.")
    print("")

    # Project data onto reduced-dimensional space
    print("Dimension reduction on example dataset.")
    pyplot.scatter(feat_norm_list['x_norm'][:, 0],
                   feat_norm_list['x_norm'][:, 1], s=80, facecolors='none',
                   edgecolors='b')
    axes = pyplot.gca()
    axes.set_xlim([-4, 3])
    axes.set_ylim([-4, 3])
    num_dim = 1
    proj_x_mat = project_data(feat_norm_list['x_norm'],
                              pca_list['left_sing_vec'], num_dim)
    print("Projection of the first example: %f" % proj_x_mat[0])
    print("(this value should be about 1.481274)")
    recov_x_mat = recover_data(proj_x_mat, pca_list['left_sing_vec'], num_dim)
    print("Approximation of the first example: %f %f" % (recov_x_mat[0, 0],
                                                         recov_x_mat[0, 1]))
    print("(this value should be about  -1.047419 -1.047419)")

    # Draw lines connecting projected points to original points
    pyplot.hold(True)
    pyplot.scatter(recov_x_mat[:, 0], recov_x_mat[:, 1], s=80,
                   facecolors='none', edgecolors='r')
    pyplot.hold(True)
    for ex_index in range(0, num_train_ex):
        return_code = draw_line(feat_norm_list['x_norm'][ex_index, :],
                                recov_x_mat[ex_index, :])
        pyplot.hold(True)
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Load and visualize face data
    print("Loading face dataset.")
    exercise_7_faces = numpy.genfromtxt("../ex7faces.txt", delimiter=",")
    return_code = display_data(exercise_7_faces[0:100, :])
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Run PCA on face data
    print("Running PCA on face dataset.")
    print("(this mght take a minute or two ...)")
    normalized_faces_list = feature_normalize(exercise_7_faces)
    faces_list = pca(normalized_faces_list['x_norm'])

    # Visualize top 36 eigenvectors for face data
    return_code = display_data(faces_list['left_sing_vec'][:, 0:36].T.conj())
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Project face data onto reduced-dimensional space
    print("Dimension reduction for face dataset.")
    num_faces_dim = 100
    proj_faces = project_data(normalized_faces_list['x_norm'],
                              faces_list['left_sing_vec'], num_faces_dim)
    print("The projected data Z has a size of:")
    print("%d %d" % (proj_faces.shape[0], proj_faces.shape[1]))
    input("Program paused. Press enter to continue.")
    print("")

    # Visualize original (normalized) and projected face data side-by-side
    print("Visualizing the projected (reduced dimension) faces.")
    recov_faces = recover_data(proj_faces, faces_list['left_sing_vec'],
                               num_faces_dim)
    fig, (ax1, ax2) = pyplot.subplots(1, 2)
    ax1.set_title('Original faces')
    return_code = display_data(normalized_faces_list['x_norm'][0:100, :], ax1)
    pyplot.hold(True)
    ax2.set_title('Recovered faces')
    return_code = display_data(recov_faces[0:100, :], ax2)
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Use PCA for visualization of high-dimensional data
    bird_small_file = open('../bird_small.png', 'rb')
    bird_small_reader = png.Reader(file=bird_small_file)
    row_count, col_count, bird_small, meta = bird_small_reader.asDirect()
    plane_count = meta['planes']
    bird_small_2d = numpy.zeros((row_count, col_count*plane_count))
    for row_index, one_boxed_row_flat_pixels in enumerate(bird_small):
        bird_small_2d[row_index, :] = one_boxed_row_flat_pixels
        bird_small_2d[row_index, :] = (1/255)*bird_small_2d[row_index,
                                                            :].astype(float)
    bird_reshape = bird_small_2d.reshape((row_count*col_count, 3))
    num_centroids = 16
    max_iter = 10
    initial_centroids = k_means_init_centroids(bird_reshape,
                                               num_centroids)
    k_m_list = run_k_means(bird_reshape, initial_centroids, max_iter, False)
    sample_idx = (
        numpy.floor(numpy.random.uniform(size=1000)*bird_reshape.shape[0]))
    palette = numpy.zeros((num_centroids+1, 3))
    for centroid_idx in range(0, num_centroids+1):
        hsv_h = centroid_idx/(num_centroids+1)
        hsv_s = 1
        hsv_v = 1
        palette[centroid_idx, :] = colors.hsv_to_rgb(numpy.r_[hsv_h, hsv_s,
                                                              hsv_v])
    fig = pyplot.figure(1)
    fig.clf()
    ax = Axes3D(fig)
    curr_colors = numpy.zeros((1000, 3))
    for data_idx in range(0, 1000):
        curr_centroid_idx = (
            k_m_list['centroid_indices'][sample_idx[data_idx]].astype(int))
        curr_colors[curr_centroid_idx, 0] = palette[curr_centroid_idx, 0]
        curr_colors[curr_centroid_idx, 1] = palette[curr_centroid_idx, 1]
        curr_colors[curr_centroid_idx, 2] = palette[curr_centroid_idx, 2]
        ax.scatter(bird_reshape[sample_idx[data_idx].astype(int), 0],
                   bird_reshape[sample_idx[data_idx].astype(int), 1],
                   bird_reshape[sample_idx[data_idx].astype(int), 2], s=80,
                   marker='o', facecolors='none',
                   edgecolors=curr_colors[curr_centroid_idx, :])
    ax.set_title('Pixel dataset plotted in 3D. Color shows centroid memberships')
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Project high-dimensional data to 2D for visualization
    normalized_bird_list = feature_normalize(bird_reshape)
    bird_list = pca(normalized_bird_list['x_norm'])
    proj_bird = project_data(normalized_bird_list['x_norm'],
                             bird_list['left_sing_vec'], 2)
    return_code = (
        plot_data_points(proj_bird[sample_idx.astype(int), :],
                         k_m_list['centroid_indices'][sample_idx.astype(int),
                                                      :], num_centroids))
    pyplot.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    pyplot.show()
    input("Program paused. Press enter to continue.")
    print("")

# Call main function
if __name__ == "__main__":
    main()
