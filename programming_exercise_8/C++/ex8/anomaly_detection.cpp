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

// AnomalyDetection class 1) implements key functions for anomaly detection
// and 2) stores relevant parameters.

#include "anomaly_detection.h"

// The number of training features should be a positive integer.
int AnomalyDetection::EstimateGaussian(const DataUnlabeled &data_unlabeled) {
  const int kNumTrainFeatures = data_unlabeled.num_features();
  assert(kNumTrainFeatures >= 1);
  data_mean_ = arma::mean(data_unlabeled.training_features()).t();
  data_variance_ = arma::var(data_unlabeled.training_features()).t();

  return 0;
}

// Assumes that training data is sampled from a Gaussian distribution.
// If "cross_val_flag" is set to 1, then this function sets the 
// cross-validation probabilities; otherwise, it sets the probabilities
// of unlabeled data.
int AnomalyDetection::MultivariateGaussian(const Data &data,\
  int cross_val_flag) {
  const int kNumTrainEx = data.num_train_ex();
  assert(kNumTrainEx >= 1);
  const int kNumTrainFeat = data.num_features();
  assert(kNumTrainFeat >= 1);
  arma::mat kVarianceMat = arma::diagmat(data_variance_);
  arma::mat kTrainFeat = data.training_features();
  for(int ex_index=0;ex_index<kNumTrainEx;ex_index++)
  {
    double kProbVecNumerator = arma::as_scalar(arma::exp(-0.5*\
      (kTrainFeat.row(ex_index)-data_mean_.t())*arma::inv(kVarianceMat)*\
      (kTrainFeat.row(ex_index)-data_mean_.t()).t()));
    double kProbVecDenominator = arma::as_scalar(pow(2.0*M_PI,\
      0.5*kNumTrainFeat)*sqrt(arma::as_scalar(arma::det(kVarianceMat))));
    if (cross_val_flag == 1) {
      data_cross_val_probs_[ex_index] = kProbVecNumerator/kProbVecDenominator;
    }
    else {
      data_probs_[ex_index] = kProbVecNumerator/kProbVecDenominator;
    }
  }

  return 0;
}

// Computes the best threshold for detecting anomalies by maximizing the
// F-score over all thresholds.
int AnomalyDetection::SelectThreshold(const Data &data) {
  const int kNumTrainEx = data.num_train_ex();
  assert(kNumTrainEx >= 1);
  double curr_f_score = 0.0;
  double kStepSize = \
    0.001*(data_cross_val_probs_.max()-data_cross_val_probs_.min());
  double curr_epsilon = data_cross_val_probs_.min();
  arma::vec predictions_vec = arma::zeros<arma::vec>(kNumTrainEx);
  arma::vec kTrainLabels = data.training_labels();
  for(int epsilon_index=0;epsilon_index<1000;epsilon_index++)
  {
    for(int ex_index=0;ex_index<kNumTrainEx;ex_index++)
    {
      predictions_vec(ex_index) = \
        (data_cross_val_probs_(ex_index) < curr_epsilon) ? 1 : 0;
    }
    int num_false_negatives = 0;
    int num_false_positives = 0;
    int num_true_positives = 0;
    for(int ex_index=0;ex_index<kNumTrainEx;ex_index++)
    {
      if ((predictions_vec(ex_index) == 1) && (kTrainLabels(ex_index) == 1)) {
        num_true_positives++;
      }
      else if ((predictions_vec(ex_index) == 1) && \
        (kTrainLabels(ex_index) == 0)) {
        num_false_positives++;
      }
      else if ((predictions_vec(ex_index) == 0) && \
        (kTrainLabels(ex_index) == 1)) {
        num_false_negatives++;
      }
    }
    if (num_true_positives > 0) {
      double precision_val = (double)num_true_positives/\
        (double)(num_true_positives+num_false_positives);
      double recall_val = (double)num_true_positives/\
        (double)(num_true_positives+num_false_negatives);
      curr_f_score = (2.0*precision_val*recall_val)/(precision_val+recall_val);
      if (curr_f_score > best_F1_) {
        best_F1_ = curr_f_score;
        best_epsilon_ = curr_epsilon;
      }
    }
    curr_epsilon += kStepSize;
  }

  return 0;
}
