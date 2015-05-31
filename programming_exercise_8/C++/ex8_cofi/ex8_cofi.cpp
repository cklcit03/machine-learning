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

// Machine Learning
// Programming Exercise 8: Recommender Systems
// Problem: Apply collaborative filtering to a dataset of movie ratings

#include "collaborative_filtering.h"
#include "load_movie_list.h"
#include "normalize_ratings.h"

int main(void) {
  printf("Loading movie ratings dataset.\n");
  const std::string kRatingsDataFileName = "../../ratingsMat.txt";
  DataUnlabeled ratings_data(kRatingsDataFileName);
  const std::string kIndicatorDataFileName = "../../indicatorMat.txt";
  DataUnlabeled indicator_data(kIndicatorDataFileName);
  printf("Average rating for movie 1 (Toy Story): %.6f / 5\n",\
    arma::sum(ratings_data.training_features().row(0))/\
    arma::sum(indicator_data.training_features().row(0)));
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Compute cost function for a subset of users, movies and features.
  const std::string kFeaturesDataFileName = "../../featuresMat.txt";
  DataUnlabeled features_data(kFeaturesDataFileName);
  const int kNumFeatures = features_data.num_features();
  const std::string kParametersDataFileName = "../../parametersMat.txt";
  DataUnlabeled parameters_data(kParametersDataFileName);
  arma::mat other_params_data;
  other_params_data.load("../../otherParams.txt",arma::csv_ascii);
  const int kSubsetNumUsers = 4;
  const int kSubsetNumMovies = 5;
  const int kSubsetNumFeatures = 3;
  const int kSubsetTotalNumFeatures = \
    kSubsetNumFeatures*(kSubsetNumUsers+kSubsetNumMovies);
  std::vector<double> theta_stack_vec(kSubsetTotalNumFeatures,0.0);
  arma::vec kFeaturesVec = \
    arma::vectorise(features_data.training_features().submat(0,0,\
      kSubsetNumMovies-1,kSubsetNumFeatures-1));
  arma::vec kParametersVec = \
    arma::vectorise(parameters_data.training_features().submat(0,0,\
      kSubsetNumUsers-1,kSubsetNumFeatures-1));
  for(int theta_index=0; theta_index<kSubsetTotalNumFeatures; theta_index++)
  {
    if (theta_index < (kSubsetNumUsers*kSubsetNumFeatures)) {
      theta_stack_vec[theta_index] = kParametersVec(theta_index);
    }
    else {
      theta_stack_vec[theta_index] = kFeaturesVec(theta_index-\
        (kSubsetNumUsers*kSubsetNumFeatures));
    }
  }
  std::vector<double> grad_vec(kSubsetTotalNumFeatures,0.0);
  CollaborativeFiltering collab_filt(theta_stack_vec,grad_vec,0.0);
  const double kInitCost = collab_filt.ComputeCost(theta_stack_vec,grad_vec,\
    ratings_data,indicator_data,kSubsetNumUsers,kSubsetNumMovies,\
    kSubsetNumFeatures);
  printf("Cost at loaded parameters: %.6f (this value should be about 22.22)\n",kInitCost);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Compute regularized cost function for a subset of users, movies and 
  // features.
  collab_filt.set_lambda(1.5);
  const double kInitCost2 = collab_filt.ComputeCost(theta_stack_vec,grad_vec,\
    ratings_data,indicator_data,kSubsetNumUsers,kSubsetNumMovies,\
    kSubsetNumFeatures);
  printf("Cost at loaded parameters (lambda = 1.5): %.6f (this value should be about 31.34)\n",kInitCost2);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Add ratings that correspond to a new user.
  std::vector<std::string> movie_list;
  const int kReturnCode = LoadMovieList(&movie_list);
  const int kNumMovies = ratings_data.num_train_ex();
  arma::vec my_ratings = arma::zeros<arma::vec>(kNumMovies,1);
  arma::vec my_indicators = arma::zeros<arma::vec>(kNumMovies,1);
  my_ratings(0) = 4;
  my_indicators(0) = 1;
  my_ratings(97) = 2;
  my_indicators(97) = 1;
  my_ratings(6) = 3;
  my_indicators(6) = 1;
  my_ratings(11) = 5;
  my_indicators(11) = 1;
  my_ratings(53) = 4;
  my_indicators(53) = 1;
  my_ratings(63) = 5;
  my_indicators(63) = 1;
  my_ratings(65) = 3;
  my_indicators(65) = 1;
  my_ratings(68) = 5;
  my_indicators(68) = 1;
  my_ratings(182) = 4;
  my_indicators(182) = 1;
  my_ratings(225) = 5;
  my_indicators(225) = 1;
  my_ratings(354) = 5;
  my_indicators(354) = 1;
  printf("New user ratings:\n");
  for(int movie_index=0; movie_index<kNumMovies; movie_index++)
  {
    if (my_ratings(movie_index) > 0) {
      printf("Rated %.0f for %s\n",arma::as_scalar(my_ratings(movie_index)),\
        movie_list[movie_index].c_str());
    }
  }
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Train collaborative filtering model.
  printf("Training collaborative filtering...\n");
  const arma::mat newRatingsMat = \
    arma::join_horiz(my_ratings,ratings_data.training_features());
  ratings_data.set_training_features(newRatingsMat);
  ratings_data.set_num_features(newRatingsMat.n_cols);
  const arma::mat newIndicatorsMat = \
    arma::join_horiz(my_indicators,indicator_data.training_features());
  indicator_data.set_training_features(newIndicatorsMat);
  indicator_data.set_num_features(newIndicatorsMat.n_cols);
  arma::vec ratings_mean = NormalizeRatings(ratings_data,indicator_data);
  const int kNumUsers = newRatingsMat.n_cols;
  collab_filt.set_lambda(10.0);
  WrapperStruct wrap_struct;
  wrap_struct.collab_filt = &collab_filt;
  wrap_struct.data_ratings = &ratings_data;
  wrap_struct.data_indicator = &indicator_data;
  wrap_struct.num_movies = kNumMovies;
  wrap_struct.num_users = kNumUsers;
  wrap_struct.num_features = kNumFeatures;
  const int kTotalNumFeatures = kNumFeatures*(kNumUsers+kNumMovies);
  nlopt::opt opt(nlopt::LD_LBFGS,kTotalNumFeatures);
  opt.set_min_objective(ComputeCostWrapper,&wrap_struct);
  opt.set_ftol_abs(1e-6);
  std::vector<double> nlopt_theta(kTotalNumFeatures,0.0);
  arma::vec nlopt_theta_vec = arma::randn(kTotalNumFeatures,1);
  for(int theta_index=0; theta_index<kTotalNumFeatures; theta_index++)
  {
    nlopt_theta[theta_index] = nlopt_theta_vec(theta_index);
  }
  double min_cost = 0.0;
  nlopt::result nlopt_result = opt.optimize(nlopt_theta,min_cost);
  arma::vec final_parameters_vec = \
    arma::zeros<arma::vec>(kNumUsers*kNumFeatures);
  arma::vec final_features_vec = \
    arma::zeros<arma::vec>(kNumMovies*kNumFeatures);
  for(int theta_index=0; theta_index<kTotalNumFeatures; theta_index++)
  {
    if (theta_index < (kNumUsers*kNumFeatures)) {
      final_parameters_vec(theta_index) = nlopt_theta[theta_index];
    }
    else {
      final_features_vec(theta_index-(kNumUsers*kNumFeatures)) = \
        nlopt_theta[theta_index];
    }
  }
  printf("Recommender system learning completed.\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Make recommendations.
  const arma::mat kFinalParametersMat = \
    arma::reshape(final_parameters_vec,kNumUsers,kNumFeatures);
  const arma::mat kFinalFeaturesMat = \
    arma::reshape(final_features_vec,kNumMovies,kNumFeatures);
  const arma::mat kPredValsMat = kFinalFeaturesMat*kFinalParametersMat.t();
  const arma::vec kMyPredVals = kPredValsMat.col(0)+ratings_mean;
  const arma::vec kSortMyPredVals = arma::sort(kMyPredVals,"descend");
  const arma::uvec kSortMyPredValsIndices = \
    arma::sort_index(kMyPredVals,"descend");
  printf("Top recommendations for you:\n");
  for(int top_movie_index=0; top_movie_index<10; top_movie_index++)
  {
    int top_movie = kSortMyPredValsIndices(top_movie_index);
    printf("Predicting rating %.1f for movie %s\n",\
      kSortMyPredVals(top_movie_index),movie_list[top_movie].c_str());
  }
  printf("Original ratings provided:\n");
  for(int movie_index=0; movie_index<kNumMovies; movie_index++)
  {
    if (my_ratings(movie_index) > 0) {
      printf("Rated %.0f for %s\n",my_ratings(movie_index),\
        movie_list[movie_index].c_str());
    }
  }

  return 0;
}
