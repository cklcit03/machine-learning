// Copyright (C) 2014  Caleb Lo
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

// DataDebug class functions manage training data, testing data and
// cross-validation data.

#include "data.h"

// The input degree should always be a positive integer.
int DataDebug::PolyFeatures(int degree_arg) {
  assert(degree_arg >= 1);
  features_poly_ = arma::zeros<arma::mat>(features_.n_rows,degree_arg);
  testing_features_poly_ = \
    arma::zeros<arma::mat>(testing_features_.n_rows,degree_arg);
  validation_features_poly_ = \
    arma::zeros<arma::mat>(validation_features_.n_rows,degree_arg);
  for(int deg_index=0; deg_index<degree_arg; deg_index++)
  {
    features_poly_.col(deg_index) = arma::pow(features_.col(1),deg_index+1);
    testing_features_poly_.col(deg_index) = \
      arma::pow(testing_features_.col(1),deg_index+1);
    validation_features_poly_.col(deg_index) = \
      arma::pow(validation_features_.col(1),deg_index+1);
  }

  return 0;
}

// The number of training, testing and cross-validation examples should always
// be a positive integer.
// The number of training, testing and cross-validation features should always
// be a positive integer.
int DataDebug::FeatureNormalize() {
  const int kNumTrainEx = features_poly_.n_rows;
  assert(kNumTrainEx >= 1);
  assert(num_test_ex_ >= 1);
  assert(num_val_ex_ >= 1);
  const int kNumFeatures = features_poly_.n_cols;
  assert(kNumFeatures >= 1);
  const int kNumTestFeatures = testing_features_poly_.n_cols;
  assert(kNumTestFeatures >= 1);
  const int kNumValFeatures = validation_features_poly_.n_cols;
  assert(kNumValFeatures >= 1);

  // Normalizes training data.
  const arma::vec mu_vec = arma::mean(features_poly_).t();
  set_mu_vec(mu_vec);
  const arma::vec sigma_vec = arma::stddev(features_poly_).t();
  set_sigma_vec(sigma_vec);
  arma::mat kTrainingFeaturesNoDummyNormalized = \
    arma::zeros<arma::mat>(kNumTrainEx,kNumFeatures);
  for(int row_index=0; row_index<kNumTrainEx;row_index++) {
    kTrainingFeaturesNoDummyNormalized.row(row_index) = \
      (features_poly_.row(row_index)-mu_vec.t())/sigma_vec.t();
  }
  const arma::mat kTrainingFeaturesNormalized = \
    arma::join_horiz(arma::ones<arma::vec>(kNumTrainEx),\
      kTrainingFeaturesNoDummyNormalized);
  set_features_normalized(kTrainingFeaturesNormalized);

  // Normalizes testing data.
  arma::mat kTestingFeaturesNoDummyNormalized = \
    arma::zeros<arma::mat>(num_test_ex_,kNumTestFeatures);
  for(int row_index=0; row_index<num_test_ex_;row_index++) {
    kTestingFeaturesNoDummyNormalized.row(row_index) = \
      (testing_features_poly_.row(row_index)-mu_vec.t())/sigma_vec.t();
  }
  const arma::mat kTestingFeaturesNormalized = \
    arma::join_horiz(arma::ones<arma::vec>(num_test_ex_),\
      kTestingFeaturesNoDummyNormalized);
  set_testing_features_normalized(kTestingFeaturesNormalized);

  // Normalizes cross-validation data.
  arma::mat kValidationFeaturesNoDummyNormalized = \
    arma::zeros<arma::mat>(num_val_ex_,kNumValFeatures);
  for(int row_index=0; row_index<num_val_ex_;row_index++) {
    kValidationFeaturesNoDummyNormalized.row(row_index) = \
      (validation_features_poly_.row(row_index)-mu_vec.t())/sigma_vec.t();
  }
  const arma::mat kValidationFeaturesNormalized = \
    arma::join_horiz(arma::ones<arma::vec>(num_val_ex_),\
      kValidationFeaturesNoDummyNormalized);
  set_validation_features_normalized(kValidationFeaturesNormalized);

  return 0;
}
