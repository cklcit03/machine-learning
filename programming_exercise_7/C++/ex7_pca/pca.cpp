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

// PCA class 1) implements key functions for principal component analysis
// and 2) stores relevant parameters.

#include "pca.h"

// Uses singular value decomposition function svd() from Armadillo.
// Thus, be sure to define the following in /path/to/armadillo/include/armadillo_bits/config.hpp:
// ARMA_USE_LAPACK
// ARMA_USE_BLAS
// ARMA_BLAS_UNDERSCORE
int PCA::Run(const DataUnlabeledNormalized &data_unlabeled_normalized) {
  const int kNumTrainEx = data_unlabeled_normalized.num_train_ex();
  assert(kNumTrainEx > 0);
  const arma::mat kTrainingFeatures = \
    data_unlabeled_normalized.training_features_normalized();
  const arma::mat kCovMat = \
    (1.0/(float)kNumTrainEx)*kTrainingFeatures.t()*kTrainingFeatures;
  const int kNumRows = kCovMat.n_rows;
  arma::mat left_sing_vec = arma::zeros<arma::mat>(kNumRows,kNumRows);
  arma::vec sing_val = arma::zeros<arma::vec>(kNumRows,1);
  arma::mat right_sing_vec = arma::zeros<arma::mat>(kNumRows,kNumRows);
  arma::svd(left_sing_vec,sing_val,right_sing_vec,kCovMat);
  this->set_left_sing_vec(left_sing_vec);
  this->set_sing_val(sing_val);

  return 0;
}

// Projects input data onto reduced-dimensional space.
int PCA::ProjectData(const DataUnlabeledNormalized &data_unlabeled_normalized,\
  int num_dim) {
  assert(num_dim > 0);
  const arma::mat kTrainingFeatures = \
    data_unlabeled_normalized.training_features_normalized();
  const arma::mat kReducedSingVec = left_sing_vec_.cols(0,num_dim-1);
  const arma::mat kMappedData = kTrainingFeatures*kReducedSingVec;
  this->set_mapped_data(kMappedData);

  return 0;
}

// Projects input data onto original space.
int PCA::RecoverData(const DataUnlabeledNormalized &data_unlabeled_normalized,\
  int num_dim) {
  assert(num_dim > 0);
  const arma::mat kReducedSingVec = left_sing_vec_.cols(0,num_dim-1);
  const arma::mat kRecoveredData = mapped_data_*kReducedSingVec.t();
  this->set_recovered_data(kRecoveredData);

  return 0;
}
