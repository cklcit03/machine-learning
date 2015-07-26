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

// GradientDescent class functions implement iterative least-squares 
// minimization.

#include "gradient_descent.h"

// The number of training examples should always be a positive integer.
double GradientDescent::ComputeCost(const DataNormalized &data) {
  const int kNumTrainEx = data.num_train_ex();
  assert(kNumTrainEx >= 1);
  const arma::mat kTrainingFeaturesNormalized = \
    data.training_features_normalized();
  const arma::vec kTrainingLabels = data.training_labels();
  const arma::vec kDiffVec = \
    kTrainingFeaturesNormalized*theta_-kTrainingLabels;
  const arma::vec kDiffVecSq = kDiffVec % kDiffVec;
  const double kJTheta = arma::as_scalar(sum(kDiffVecSq))/(2.0*kNumTrainEx);

  return kJTheta;
}

// The step size should be chosen carefully to guarantee convergence given a 
// reasonable number of computations.
int GradientDescent::RunGradientDescent(const DataNormalized &data) {
  assert(num_iters_ >= 1);
  const int kNumTrainEx = data.num_train_ex();
  assert(kNumTrainEx >= 1);

  // For this exercise, there should be at least two training features.
  const int kNumFeatures = data.num_features();
  assert(kNumFeatures >= 2);

  const arma::mat kTrainingFeaturesNormalized = \
    data.training_features_normalized();
  const arma::vec kTrainingLabels = data.training_labels();

  double *j_theta_array = new double[num_iters_];

  // Recall that we are trying to minimize the following cost function:
  // ((training features) * (current weights) - (training labels))^2
  // Each iteration of this algorithm updates the weights as a scaled 
  // version of the gradient of this cost function.
  // This gradient is computed with respect to the weights.
  // Thus, each iteration of gradient descent performs the following:
  // (update) = (step size) * ((training features) * (current weights) - (training labels)) * (training features)
  // (new weights) = (current weights) - (update)
  for(int theta_index=0; theta_index<num_iters_; theta_index++)
  {
    const arma::vec kDiffVec = \
      kTrainingFeaturesNormalized*theta_-kTrainingLabels;
    arma::mat kDiffVecTimesTrainFeat = \
      kDiffVec % kTrainingFeaturesNormalized.col(0);
    for(int feature_index=1; feature_index<=kNumFeatures; feature_index++)
    {
      kDiffVecTimesTrainFeat = join_rows(kDiffVecTimesTrainFeat,\
        kDiffVec % kTrainingFeaturesNormalized.col(feature_index));
    }
    const arma::vec kThetaNew = theta_-alpha_*(1/(float)kNumTrainEx)*\
      (sum(kDiffVecTimesTrainFeat)).t();
    j_theta_array[theta_index] = ComputeCost(data);
    set_theta(kThetaNew);
  }

  delete [] j_theta_array;

  return 0;
}
