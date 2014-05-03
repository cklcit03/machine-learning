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

// Data class functions manage training data.

#include "data.h"

// The number of training examples should always be a positive integer.
int DataNormalized::FeatureNormalize() {
  const int kNumFeatures = num_features();
  assert(kNumFeatures > 0);
  const int kNumTrainEx = num_train_ex();
  assert(kNumTrainEx > 0);

  // Do not include dummy feature when normalizing.
  const arma::mat kTrainingFeaturesNoDummy = \
    training_features().cols(1,kNumFeatures);

  const arma::vec mu_vec = \
    arma::mean(kTrainingFeaturesNoDummy.cols(0,kNumFeatures-1)).t();
  set_mu_vec(mu_vec);
  const arma::vec sigma_vec = \
    arma::stddev(kTrainingFeaturesNoDummy.cols(0,kNumFeatures-1)).t();
  set_sigma_vec(sigma_vec);
  arma::mat kTrainingFeaturesNoDummyNormalized = \
    arma::zeros<arma::mat>(kNumTrainEx,kNumFeatures);
  for(int row_index=0; row_index<kNumTrainEx;row_index++) {
    kTrainingFeaturesNoDummyNormalized.row(row_index) = \
      (kTrainingFeaturesNoDummy.row(row_index)-mu_vec.t())/sigma_vec.t();
  }
  const arma::mat kTrainingFeaturesNormalized = \
    arma::join_horiz(arma::ones<arma::vec>(kNumTrainEx),\
      kTrainingFeaturesNoDummyNormalized);
  set_training_features_normalized(kTrainingFeaturesNormalized);

  return 0;
}
