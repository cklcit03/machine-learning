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
# Programming Exercise 8: Recommender Systems
# Problem: Apply collaborative filtering to a dataset of movie ratings
from matplotlib import cm
from matplotlib import pyplot
from scipy.optimize import fmin_ncg
import numpy


# Current iteration of fmin_ncg
n_feval = 1


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def callback_fmin_ncg(Xi):
    """ Displays current iteration of fmin_ncg.

    Args:
      Xi: Current parameter vector.

    Returns:
      None.
    """
    global n_feval
    print("n_feval = %d" % n_feval)
    n_feval += 1


def compute_cost(theta, y, R, num_train_ex, lamb, num_users, num_movies,
                 num_features):
    """ Computes regularized cost function J(\theta).

    Args:
      theta: Vector of parameters for regularized linear regression.
      y: Matrix of movie ratings.
      R: Binary-valued indicator matrix, where the (i,j)-th entry is 1 only if 
         user j has rated movie i.
      num_train_ex: Number of training examples.
      lamb: Regularization parameter.
      num_users: Number of users.
      num_movies: Number of movies.
      num_features: Number of features for each movie (or user).

    Returns:
      j_theta_reg: Regularized linear regression cost.

    Raises:
      An error occurs if the number of training examples is 0.
      An error occurs if the total number of features is 0.
    """
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    total_num_features = num_features*(num_users+num_movies)
    if (total_num_features == 0): raise Error('total_num_features = 0')
    theta = numpy.reshape(theta, (total_num_features, 1), order='F')
    params_vec = theta[0:(num_users*num_features), :]
    params_vec_sq = numpy.power(params_vec, 2)
    features_vec = theta[(num_users*num_features):total_num_features, :]
    features_vec_sq = numpy.power(features_vec, 2)
    params_mat = numpy.reshape(params_vec, (num_users, num_features),
                               order='F')
    ft_mat = numpy.reshape(features_vec, (num_movies, num_features), order='F')
    y_mat = (
        numpy.multiply((numpy.ones((num_users,
                                    num_movies))-numpy.transpose(R)),
                       (numpy.dot(params_mat,
                                  numpy.transpose(ft_mat))))+numpy.transpose(y))
    cost_function_mat = numpy.dot(params_mat, numpy.transpose(ft_mat))-y_mat
    cost_function_vec = cost_function_mat[:, 0]
    for column_index in range(1, cost_function_mat.shape[1]):
        cost_function_vec = numpy.vstack((cost_function_vec,
                                          cost_function_mat[:, column_index]))
    j_theta = (1/2)*numpy.sum(numpy.power(cost_function_vec, 2))
    j_theta_reg = (
        j_theta+(lamb/2)*(numpy.sum(params_vec_sq)+numpy.sum(features_vec_sq)))
    return j_theta_reg


def compute_gradient(theta, y, R, num_train_ex, lamb, num_users, num_movies,
                     num_features):
    """ Computes gradient of regularized cost function J(\theta).

    Args:
      theta: Vector of parameters for regularized linear regression.
      y: Matrix of movie ratings.
      R: Binary-valued indicator matrix, where the (i,j)-th entry is 1 only if 
         user j has rated movie i.
      num_train_ex: Number of training examples.
      lamb: Regularization parameter.
      num_users: Number of users.
      num_movies: Number of movies.
      num_features: Number of features for each movie (or user).

    Returns:
      grad_array_reg_flat: Vector of regularized linear regression gradients
                           (one per feature).

    Raises:
      An error occurs if the number of training examples is 0.
      An error occurs if the total number of features is 0.
    """
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    total_num_features = num_features*(num_users+num_movies)
    if (total_num_features == 0): raise Error('total_num_features = 0')
    theta = numpy.reshape(theta, (total_num_features, 1), order='F')
    params_vec = theta[0:(num_users*num_features), :]
    params_vec_sq = numpy.power(params_vec, 2)
    features_vec = theta[(num_users*num_features):total_num_features, :]
    features_vec_sq = numpy.power(features_vec, 2)
    params_mat = numpy.reshape(params_vec, (num_users, num_features), order='F')
    ft_mat = numpy.reshape(features_vec, (num_movies, num_features), order='F')
    y_mat = (
        numpy.multiply((numpy.ones((num_users,
                                    num_movies))-numpy.transpose(R)),
                       (numpy.dot(params_mat,
                                  numpy.transpose(ft_mat))))+numpy.transpose(y))
    diff_mat = numpy.transpose(numpy.dot(params_mat,
                                         numpy.transpose(ft_mat))-y_mat)
    grad_params_array = numpy.zeros((num_users*num_features, 1))
    grad_params_array_reg = numpy.zeros((num_users*num_features, 1))
    for grad_index in range(0, num_users*num_features):
        user_index = 1+numpy.mod(grad_index, num_users)
        ft_index = 1+((grad_index-numpy.mod(grad_index, num_users))/num_users)
        grad_params_array[grad_index] = (
            numpy.sum(numpy.multiply(diff_mat[:, user_index-1],
                                     ft_mat[:, ft_index-1])))
        grad_params_array_reg[grad_index] = (
            grad_params_array[grad_index]+lamb*params_vec[grad_index])
    grad_features_array = numpy.zeros((num_movies*num_features, 1))
    grad_features_array_reg = numpy.zeros((num_movies*num_features, 1))
    for grad_index in range(0, num_movies*num_features):
        movie_index = 1+numpy.mod(grad_index, num_movies)
        ft_index = 1+((grad_index-numpy.mod(grad_index, num_movies))/num_movies)
        grad_features_array[grad_index] = (
            numpy.sum(numpy.multiply(diff_mat[movie_index-1, :],
                                     numpy.transpose(params_mat[:,
                                                                ft_index-1]))))
        grad_features_array_reg[grad_index] = (
            grad_features_array[grad_index]+lamb*features_vec[grad_index])
    grad_array_reg = numpy.zeros((total_num_features, 1))
    grad_array_reg[0:(num_users*num_features), :] = grad_params_array_reg
    grad_array_reg[(num_users*num_features):total_num_features, :] = (
        grad_features_array_reg)
    grad_array_reg_flat = numpy.ndarray.flatten(grad_array_reg)
    return grad_array_reg_flat


def compute_cost_grad_list(y, R, theta, lamb, num_users, num_movies,
                           num_features):
    """ Aggregates computed cost and gradient.

    Args:
      y: Matrix of movie ratings.
      R: Binary-valued indicator matrix, where the (i,j)-th entry is 1 only if 
         user j has rated movie i.
      theta: Vector of parameters for regularized linear regression.
      lamb: Regularization parameter.
      num_users: Number of users.
      num_movies: Number of movies.
      num_features: Number of features for each movie (or user).

    Returns:
      return_list: List of two objects.
                   j_theta_reg: Updated regularized linear regression cost.
                   grad_array_reg: Updated vector of regularized linear 
                                   regression gradients (one per feature).
    """
    num_train_ex = y.shape[0]
    j_theta_reg = compute_cost(theta, y, R, num_train_ex, lamb, num_users,
                               num_movies, num_features)
    grad_array_reg_flat = compute_gradient(theta, y, R, num_train_ex, lamb,
                                           num_users, num_movies, num_features)
    total_num_features = num_features*(num_users+num_movies)
    grad_array_reg = numpy.reshape(grad_array_reg_flat, (total_num_features, 1),
                                   order='F')
    return_list = {'j_theta_reg': j_theta_reg, 'grad_array_reg': grad_array_reg}
    return return_list


def load_movie_list():
    """ Returns list of movies.

    Args:
      None.

    Returns:
      movie_list: Vector of movies, where each entry consists of a movie ID, the
                  name of that movie, and the year that it was released.
    """
    movie_ids_file = open('../movie_ids.txt', 'r')
    movie_list = []
    for line in movie_ids_file:
        token_count = 0
        movie_id = ""
        for token in line.split():
            if (token_count > 0):
                movie_id += token
                movie_id += " "
            token_count = token_count + 1
        movie_id = movie_id.lstrip(' ')
        movie_id = movie_id.rstrip(' ')
        movie_list.append(movie_id)
    movie_ids_file.close()
    return movie_list


def normalize_ratings(Y, R):
    """ Normalizes movie ratings.

    Args:
      Y: Matrix of movie ratings.
      R: Binary-valued indicator matrix, where the (i,j)-th entry is 1 only if 
         user j has rated movie i.

    Returns:
      return_list: List of two objects.
                   y_mean: Vector of mean movie ratings.
                   y_norm: Normalized matrix of movie ratings, where
                           normalization is performed using entries of y_mean.

    Raises:
      An error occurs if the number of movies is 0.
    """
    num_movies = Y.shape[0]
    if (num_movies == 0): raise Error('num_movies = 0')
    num_users = Y.shape[1]
    y_mean = numpy.zeros((num_movies, 1))
    y_norm = numpy.zeros((num_movies, num_users))
    for movie_index in range(0, num_movies):
        rated_users = numpy.where(R[movie_index, :] == 1)
        y_mean[movie_index, ] = numpy.sum(Y[movie_index,
                                            :])/numpy.sum(R[movie_index, :])
        y_norm[movie_index, rated_users] = Y[movie_index,
                                             rated_users]-y_mean[movie_index, :]
    return_list = {'y_mean': y_mean, 'y_norm': y_norm}
    return return_list


def main():
    """ Main function

    Raises:
      An error occurs if the number of movies is 0.
    """
    print("Loading movie ratings dataset.")
    ratings_data = numpy.genfromtxt("../ratingsMat.txt", delimiter=",")
    num_movies = ratings_data.shape[0]
    if (num_movies == 0): raise Error('num_movies = 0')
    num_users = ratings_data.shape[1]
    ratings_mat = ratings_data[:, 0:num_users]
    indicator_data = numpy.genfromtxt("../indicatorMat.txt", delimiter=",")
    indicator_mat = indicator_data[:, 0:num_users]
    print("Average rating for movie 1 (Toy Story): %.6f / 5" %
          numpy.mean(ratings_mat[0, numpy.where(indicator_mat[0, :] == 1)]))
    pyplot.imshow(numpy.transpose(ratings_mat), cmap=cm.coolwarm,
                  origin='lower')
    pyplot.ylabel('Movies', fontsize=18)
    pyplot.xlabel('Users', fontsize=18)
    pyplot.gca().xaxis.set_major_locator(pyplot.NullLocator())
    pyplot.gca().yaxis.set_major_locator(pyplot.NullLocator())
    pyplot.show()
    input("Program paused. Press enter to continue.")

    # Compute cost function for a subset of users, movies and features
    features_data = numpy.genfromtxt("../featuresMat.txt", delimiter=",")
    num_features = features_data.shape[1]
    ft_mat = features_data[:, 0:num_features]
    parameters_data = numpy.genfromtxt("../parametersMat.txt", delimiter=",")
    parameters_mat = parameters_data[:, 0:num_features]
    other_params_data = numpy.genfromtxt("../otherParams.txt", delimiter=",")
    subset_num_users = 4
    subset_num_movies = 5
    subset_num_features = 3
    subset_features_mat = ft_mat[0:subset_num_movies, 0:subset_num_features]
    subset_parameters_mat = parameters_mat[0:subset_num_users,
                                           0:subset_num_features]
    subset_ratings_mat = ratings_mat[0:subset_num_movies, 0:subset_num_users]
    subset_indicator_mat = indicator_mat[0:subset_num_movies,
                                         0:subset_num_users]
    parameters_vec = subset_parameters_mat[:, 0]
    for ft_index in range(1, subset_num_features):
        parameters_vec = numpy.hstack((parameters_vec,
                                       subset_parameters_mat[:, ft_index]))
    parameters_vec = numpy.reshape(parameters_vec,
                                   (subset_num_users*subset_num_features, 1),
                                   order='F')
    features_vec = subset_features_mat[:, 0]
    for ft_index in range(1, subset_num_features):
        features_vec = numpy.hstack((features_vec,
                                     subset_features_mat[:, ft_index]))
    features_vec = numpy.reshape(features_vec,
                                 (subset_num_movies*subset_num_features, 1),
                                 order='F')
    subset_total_num_features = (
        subset_num_features*(subset_num_users+subset_num_movies))
    theta_vec = numpy.zeros((subset_total_num_features, 1))
    theta_vec[0:(subset_num_users*subset_num_features), :] = parameters_vec
    theta_vec[(subset_num_users*subset_num_features):subset_total_num_features,
              :] = features_vec
    lamb = 0
    init_compute_cost_list = compute_cost_grad_list(subset_ratings_mat,
                                                    subset_indicator_mat,
                                                    theta_vec, lamb,
                                                    subset_num_users,
                                                    subset_num_movies,
                                                    subset_num_features)
    print("Cost at loaded parameters: %.6f (this value should be about 22.22)" %
          init_compute_cost_list['j_theta_reg'])
    input("Program paused. Press enter to continue.")

    # Compute regularized cost function for a subset of users, movies and features
    lamb = 1.5
    init_compute_cost_list = compute_cost_grad_list(subset_ratings_mat,
                                                    subset_indicator_mat,
                                                    theta_vec, lamb,
                                                    subset_num_users,
                                                    subset_num_movies,
                                                    subset_num_features)
    print("Cost at loaded parameters (lambda = 1.5): %.6f (this value should be about 31.34)" %
          init_compute_cost_list['j_theta_reg'])
    input("Program paused. Press enter to continue.")

    # Add ratings that correspond to a new user
    movie_list = load_movie_list()
    my_ratings = numpy.zeros((num_movies, 1))
    my_ratings[0, :] = 4
    my_ratings[97, :] = 2
    my_ratings[6, :] = 3
    my_ratings[11, :] = 5
    my_ratings[53, :] = 4
    my_ratings[63, :] = 5
    my_ratings[65, :] = 3
    my_ratings[68, :] = 5
    my_ratings[182, :] = 4
    my_ratings[225, :] = 5
    my_ratings[354, :] = 5
    print("New user ratings:")
    for movie_index in range(0, num_movies):
        if (my_ratings[movie_index, :] > 0):
            print("Rated %d for %s" % (my_ratings[movie_index, :],
                                       movie_list[movie_index]))
    input("Program paused. Press enter to continue.")

    # Train collaborative filtering model
    print("Training collaborative filtering...")
    ratings_mat = numpy.c_[my_ratings, ratings_mat]
    my_indicators = (my_ratings != 0)
    indicator_mat = numpy.c_[my_indicators, indicator_mat]
    normalize_ratings_list = normalize_ratings(ratings_mat, indicator_mat)
    num_users = ratings_mat.shape[1]
    numpy.random.seed(1)
    parameters_vec = numpy.random.normal(size=(num_users*num_features, 1))
    features_vec = numpy.random.normal(size=(num_movies*num_features, 1))
    total_num_features = num_features*(num_users+num_movies)
    theta_vec = numpy.zeros((total_num_features, 1))
    theta_vec[0:(num_users*num_features), :] = parameters_vec
    theta_vec[(num_users*num_features):total_num_features, :] = features_vec
    theta_vec_flat = numpy.ndarray.flatten(theta_vec)
    lamb = 10
    fmin_ncg_out = fmin_ncg(compute_cost, theta_vec_flat,
                            args=(ratings_mat, indicator_mat, num_movies, lamb,
                                  num_users, num_movies, num_features),
                            fprime=compute_gradient, callback=callback_fmin_ncg,
                            maxiter=100, full_output=1)
    theta_opt = numpy.reshape(fmin_ncg_out[0], (total_num_features, 1),
                              order='F')
    final_parameters_vec = theta_opt[0:(num_users*num_features), :]
    final_features_vec = theta_opt[(num_users*num_features):total_num_features,
                                   :]
    print("Recommender system learning completed.")
    input("Program paused. Press enter to continue.")

    # Make recommendations
    final_parameters_mat = numpy.reshape(final_parameters_vec,
                                         (num_users, num_features), order='F')
    final_features_mat = numpy.reshape(final_features_vec,
                                       (num_movies, num_features), order='F')
    pred_vals = numpy.dot(final_features_mat,
                          numpy.transpose(final_parameters_mat))
    my_pred_vals = numpy.reshape(pred_vals[:, 0], (num_movies, 1), order='F')
    my_pred_vals += normalize_ratings_list['y_mean']
    sort_my_pred_vals = numpy.sort(my_pred_vals, axis=None)[::-1]
    sort_my_pred_indices = numpy.reshape(numpy.argsort(my_pred_vals,
                                                       axis=None)[::-1],
                                         (num_movies, 1), order='F')
    print("Top recommendations for you:")
    for top_movie_index in range(0, 10):
        top_movie = sort_my_pred_indices[top_movie_index]
        print("Predicting rating %.1f for movie %s" %
              (sort_my_pred_vals[top_movie_index], movie_list[top_movie]))
    print("Original ratings provided:")
    for movie_index in range(0, num_movies):
        if (my_ratings[movie_index, :] > 0):
            print("Rated %d for %s" % (my_ratings[movie_index, :],
                                       movie_list[movie_index]))

# Call main function
if __name__ == "__main__":
    main()
