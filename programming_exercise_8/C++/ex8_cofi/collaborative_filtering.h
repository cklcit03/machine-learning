// Copyright (C) 2015  Caleb Lo
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.

// CollaborativeFiltering class 1) implements key functions for collaborative
// filtering and 2) stores relevant parameters.

#ifndef MACHINE_LEARNING_PROGRAMMING_EXERCISE_8_EX8_COFI_COLLABORATIVE_FILTERING_H_
#define MACHINE_LEARNING_PROGRAMMING_EXERCISE_8_EX8_COFI_COLLABORATIVE_FILTERING_H_

#include <assert.h>
#include <string>

#include "armadillo"
#include "nlopt.hpp"

#include "data.h"

// Implements key functions for collaborative filtering and stores relevant 
// parameters.
// Sample usage:
// CollaborativeFiltering collab_filt(kIterations,theta_vec,grad_vec,kLambda,kNumUsers,kNumMovies,kNumFeatures);
// const double kInitReturnCode = collab_filt.ComputeGradient(ratings_data,indicator_data);
class CollaborativeFiltering
{
 public:
  // Sets default values for algorithm parameters.
  CollaborativeFiltering() {
    theta_.zeros(3,1);
    gradient_.zeros(3,1);
    lambda_ = 0.0;
  }

  // Sets values for algorithm parameters.
  // "theta_arg" corresponds to an initial guess of the weights.
  // "gradient_arg" corresponds to an initial guess of the gradient.
  // "lambda_arg" corresponds to the regularization parameter
  CollaborativeFiltering(arma::vec theta_arg,arma::vec gradient_arg,\
    double lambda_arg) : theta_(theta_arg),gradient_(gradient_arg),\
    lambda_(lambda_arg) {}

  ~CollaborativeFiltering() {}

  // Computes cost function given ratings data in "ratings_data", indicator 
  // data in "indicator_data", current weights in theta_ and current 
  // regularization parameter lambda_.
  // "num_users", "num_movies" and "num_features" denote the number of users,
  // movies and features, respectively.
  // Consider the parameter vectors over all users and the feature vectors over
  // all movies; theta_ is obtained by vectorizing all of these vectors.
  // Cost function is a sum over all (movie i, user j) pairs where user j has
  // rated movie i.
  // The (i, j)-th term in this sum is (w/o regularization): 
  // (1/2) * ((inner product between parameter vector for user j and feature
  // vector for movie i) - (rating for movie i by user j))^2
  // Sums all of these terms to obtain cost function.
  // To this cost, adds following two terms:
  // 1) (lambda_ / 2) * (sum over all users of (square of L2 norm of parameter
  // vector for user j))
  // 2) (lambda_ / 2) * (sum over all movies of (square of L2 norm of feature
  // vector for movie i))
  double ComputeCost(const std::vector<double> &opt_param,
    std::vector<double> &grad,const DataUnlabeled &ratings_data,
    const DataUnlabeled &indicator_data,int num_users,int num_movies,
    int num_features);

  // Computes gradient given ratings data in "ratings_data", indicator 
  // data in "indicator_data", current weights in theta_ and current 
  // regularization parameter lambda_.
  // "num_users", "num_movies" and "num_features" denote the number of users,
  // movies and features, respectively.
  // Gradient term (for feature vector i and training feature k) is
  // a sum over all users j where user j has rated movie i.
  // Each term in this sum is (w/o regularization):
  // ((inner product between parameter vector for user j and feature
  // vector for movie i) - (rating for movie i by user j)) * 
  // (k-th term of parameter vector for user j)
  // To this gradient, adds following term:
  // lambda_ * (k-th term of feature vector for movie i)
  // Gradient term (for parameter vector j and training feature k) is
  // a sum over all movies i where user j has rated movie i.
  // Each term in this sum is (w/o regularization):
  // ((inner product between parameter vector for user j and feature
  // vector for movie i) - (rating for movie i by user j)) * 
  // (k-th term of feature vector for movie i)
  // To this gradient, adds following term:
  // lambda_ * (k-th term of parameter vector for movie j)
  int ComputeGradient(const DataUnlabeled &ratings_data,
    const DataUnlabeled &indicator_data,int num_users,int num_movies,
    int num_features);

  inline virtual arma::vec theta() const {
    return theta_;
  }

  inline virtual arma::vec gradient() const {
    return gradient_;
  }

  inline double lambda() const {
    return lambda_;
  }

  inline virtual int set_theta(arma::vec theta_arg) {
    theta_ = theta_arg;

    return 0;
  }

  inline virtual int set_gradient(arma::vec gradient_arg) {
    gradient_ = gradient_arg;

    return 0;
  }

  inline int set_lambda(double lambda_arg) {
    lambda_ = lambda_arg;

    return 0;
  }

 private:
  // Current weights.
  arma::vec theta_;

  // Current gradient.
  arma::vec gradient_;

  // Regularization parameter for linear regression.
  double lambda_;

  DISALLOW_COPY_AND_ASSIGN(CollaborativeFiltering);
};

// Defines a struct that contains an instance of CollaborativeFiltering 
// class, two instances of DataUnlabeled class, and current number of users,
// movies, and features.  This struct will be passed as "void_data" to 
// ComputeCostWrapper.
struct WrapperStruct {
  CollaborativeFiltering *collab_filt;
  DataUnlabeled *data_ratings;
  DataUnlabeled *data_indicator;
  int num_users;
  int num_movies;
  int num_features;
};

// nlopt requires a wrapper function.  A WrapperStruct is contained in
// void_data, and it is unpacked in this wrapper function.
// This wrapper function calls ComputeCost to update "opt_param" and "grad".
double ComputeCostWrapper(const std::vector<double> &opt_param,
  std::vector<double> &grad,void *void_data);

#endif	// MACHINE_LEARNING_PROGRAMMING_EXERCISE_8_EX8_COFI_COLLABORATIVE_FILTERING_H_
