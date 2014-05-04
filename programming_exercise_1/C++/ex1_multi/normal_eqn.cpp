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

// Defines function that computes normal equations.

#include "normal_eqn.h"

// Uses pseudoinverse function pinv() from Armadillo.
// Thus, be sure to define the following in /path/to/armadillo/include/armadillo_bits/config.hpp:
// ARMA_USE_LAPACK
// ARMA_USE_BLAS
// ARMA_BLAS_UNDERSCORE
arma::vec NormalEqn(const DataNormalized &data) {
  const arma::mat kTrainingFeatures = data.training_features();
  const arma::vec kTrainingLabels = data.training_labels();
  const arma::vec thetaNormal = \
    pinv(kTrainingFeatures.t()*kTrainingFeatures)*\
    kTrainingFeatures.t()*kTrainingLabels;

  return thetaNormal;
}
