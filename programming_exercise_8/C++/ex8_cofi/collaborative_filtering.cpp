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

#include "collaborative_filtering.h"

// Arguments "opt_param" and "grad" are updated by nlopt.
// "opt_param" corresponds to the optimization parameters.
// "grad" corresponds to the gradient.
double CollaborativeFiltering::ComputeCost(
  const std::vector<double> &opt_param,std::vector<double> &grad,
  const DataUnlabeled &ratings_data,const DataUnlabeled &indicator_data,
  int num_users,int num_movies,int num_features) {
  assert(num_users >= 1);
  assert(num_movies >= 1);
  assert(num_features >= 1);
  const int kTotalNumFeatures = num_features*(num_users+num_movies);
  arma::vec nlopt_theta = arma::zeros<arma::vec>(kTotalNumFeatures,1);
  arma::mat features_cvec = arma::zeros<arma::mat>(num_movies*num_features,1);
  arma::mat params_cvec = arma::zeros<arma::mat>(num_users*num_features,1);

  // Uses current value of "theta" from nlopt.
  for(int feature_index=0; feature_index<kTotalNumFeatures; feature_index++)
  {
    nlopt_theta(feature_index) = opt_param[feature_index];
    if (feature_index < (num_users*num_features)) {
      params_cvec(feature_index) = opt_param[feature_index];
    }
    else {
      features_cvec(feature_index-(num_users*num_features)) = \
        opt_param[feature_index];
    }
  }
  set_theta(nlopt_theta);
  arma::mat features_cvec_squared = arma::square(features_cvec);
  arma::mat params_cvec_squared = arma::square(params_cvec);
  arma::mat params_mat = arma::reshape(params_cvec,num_users,num_features);
  arma::mat features_mat = \
    arma::reshape(features_cvec,num_movies,num_features);

  // Computes cost function given current value of "theta".
  const arma::mat kSubsetIndicatorMat = \
    indicator_data.training_features().submat(0,0,num_movies-1,num_users-1);
  const arma::mat kSubsetRatingsMat = \
    ratings_data.training_features().submat(0,0,num_movies-1,num_users-1);
  const arma::mat kYMat = \
    (arma::ones<arma::mat>(num_users,num_movies)-kSubsetIndicatorMat.t()) % \
    (params_mat*features_mat.t())+kSubsetRatingsMat.t();
  const arma::mat kCostFunctionMat = params_mat*features_mat.t()-kYMat;
  const arma::vec kCostFunctionVec = arma::vectorise(kCostFunctionMat);
  const double kJTheta = 0.5*arma::as_scalar(sum(square(kCostFunctionVec)));

  // Adds regularization term.
  const double kRegTerm = 0.5*lambda_*\
    (as_scalar(sum(params_cvec_squared))+\
    as_scalar(sum(features_cvec_squared)));
  const double kJThetaReg = kJTheta+kRegTerm;

  // Updates "grad" for nlopt.
  const int kReturnCode = this->ComputeGradient(ratings_data,indicator_data,\
    num_users,num_movies,num_features);
  const arma::vec kCurrentGradient = gradient();
  for(int feature_index=0; feature_index<kTotalNumFeatures; feature_index++)
  {
    grad[feature_index] = kCurrentGradient(feature_index);
  }

  return kJThetaReg;
}

// The number of users should always be a positive integer.
// The number of movies should always be a positive integer.
// The number of features should always be a positive integer.
int CollaborativeFiltering::ComputeGradient(const DataUnlabeled &ratings_data,
  const DataUnlabeled &indicator_data,int num_users,int num_movies,
  int num_features) {
  assert(num_users >= 1);
  assert(num_movies >= 1);
  assert(num_features >= 1);
  const int kTotalNumFeatures = num_features*(num_users+num_movies);
  const arma::vec kCurrentTheta = theta();
  arma::mat features_cvec = arma::zeros<arma::mat>(num_movies*num_features,1);
  arma::mat params_cvec = arma::zeros<arma::mat>(num_users*num_features,1);
  for(int feature_index=0; feature_index<kTotalNumFeatures; feature_index++)
  {
    if (feature_index < (num_users*num_features)) {
      params_cvec(feature_index) = kCurrentTheta[feature_index];
    }
    else {
      features_cvec(feature_index-(num_users*num_features)) = \
        kCurrentTheta[feature_index];
    }
  }
  arma::mat features_cvec_squared = arma::square(features_cvec);
  arma::mat params_cvec_squared = arma::square(params_cvec);
  arma::mat params_mat = arma::reshape(params_cvec,num_users,num_features);
  arma::mat features_mat = \
    arma::reshape(features_cvec,num_movies,num_features);
  const arma::mat kSubsetIndicatorMat = \
    indicator_data.training_features().submat(0,0,num_movies-1,num_users-1);
  const arma::mat kSubsetRatingsMat = \
    ratings_data.training_features().submat(0,0,num_movies-1,num_users-1);
  const arma::mat kYMat = \
    (arma::ones<arma::mat>(num_users,num_movies)-kSubsetIndicatorMat.t()) % \
    (params_mat*features_mat.t())+kSubsetRatingsMat.t();
  const arma::mat kDiffMat = (params_mat*features_mat.t()-kYMat).t();
  arma::mat grad_params_arr = arma::zeros<arma::mat>(num_users*num_features,1);
  arma::mat grad_params_arr_reg = \
    arma::zeros<arma::mat>(num_users*num_features,1);
  for(int grad_index=0; grad_index<(num_users*num_features); grad_index++)
  {
    int user_index = 1+(grad_index % num_users);
    int feature_index = 1+((grad_index-(grad_index % num_users))/num_users);
    grad_params_arr(grad_index) = \
      arma::as_scalar(sum(kDiffMat.col(user_index-1) % \
      features_mat.col(feature_index-1)));
    grad_params_arr_reg(grad_index) = \
      grad_params_arr(grad_index)+lambda_*params_cvec(grad_index);
  }
  arma::mat grad_features_arr = \
    arma::zeros<arma::mat>(num_movies*num_features,1);
  arma::mat grad_features_arr_reg = \
    arma::zeros<arma::mat>(num_movies*num_features,1);
  for(int grad_index=0; grad_index<(num_movies*num_features); grad_index++)
  {
    int movie_index = 1+(grad_index % num_movies);
    int feature_index = 1+((grad_index-(grad_index % num_movies))/num_movies);
    grad_features_arr(grad_index) = \
      arma::as_scalar(sum(kDiffMat.row(movie_index-1) % \
      (params_mat.col(feature_index-1)).t()));
    grad_features_arr_reg(grad_index) = \
      grad_features_arr(grad_index)+lambda_*features_cvec(grad_index);
  }
  arma::vec gradient_array_reg = arma::zeros<arma::vec>(kTotalNumFeatures);
  gradient_array_reg.subvec(0,(num_users*num_features-1)) = \
    grad_params_arr_reg;
  gradient_array_reg.subvec(num_users*num_features,kTotalNumFeatures-1) = \
    grad_features_arr_reg;
  set_gradient(gradient_array_reg);

  return 0;
}

// Unpacks WrapperStruct to obtain an instance of CollaborativeFiltering, two 
// instances of DataUnlabeled (one for ratings data and the other for
// indicator data), and current number of users, movies and features.
double ComputeCostWrapper(const std::vector<double> &opt_param,
    std::vector<double> &grad,void *void_data) {
  WrapperStruct *wrap_struct = static_cast<WrapperStruct *>(void_data);
  CollaborativeFiltering *collab_filt = wrap_struct->collab_filt;
  DataUnlabeled *data_ratings = wrap_struct->data_ratings;
  DataUnlabeled *data_indicator = wrap_struct->data_indicator;
  int num_users = wrap_struct->num_users;
  int num_movies = wrap_struct->num_movies;
  int num_features = wrap_struct->num_features;

  return collab_filt->ComputeCost(opt_param,grad,*data_ratings,
    *data_indicator,num_users,num_movies,num_features);
}
