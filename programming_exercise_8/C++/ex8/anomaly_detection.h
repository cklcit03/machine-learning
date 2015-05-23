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

#ifndef ANOMALY_DETECTION_H_
#define ANOMALY_DETECTION_H_

#include <assert.h>
#include <math.h>
#include <string>

#include "data.h"

#define M_PI 3.14159265358979323846

// Implements key functions for anomaly detection and stores relevant 
// parameters.
// Sample usage:
// AnomalyDetection anom_detect(training_data);
// const int kReturnCode = anom_detect.EstimateGaussian();
class AnomalyDetection
{
 public:
  // Sets default values for algorithm parameters.
  AnomalyDetection() {
    data_mean_ = arma::zeros<arma::vec>(1);
    data_variance_ = arma::zeros<arma::vec>(1);
    data_probs_ = arma::zeros<arma::vec>(1);
    data_cross_val_probs_ = arma::zeros<arma::vec>(1);
    best_F1_ = 0.0;
    best_epsilon_ = 0.0;
  }

  // Sets values for algorithm parameters.
  // "data_unlabeled" corresponds to the unlabeled training data.
  // "data" corresponds to labeled cross-validation data.
  AnomalyDetection(const DataUnlabeled &data_unlabeled,const Data &data) {

    // Sets the mean of the training data.
    data_mean_ = arma::zeros<arma::vec>(data_unlabeled.num_features());

    // Sets the variance of the training data.
    data_variance_ = arma::zeros<arma::vec>(data_unlabeled.num_features());

    // Sets the probability of each training example.
    data_probs_ = arma::zeros<arma::vec>(data_unlabeled.num_train_ex());

	// Sets the probability of each cross-validation example.
	data_cross_val_probs_ = arma::zeros<arma::vec>(data.num_train_ex());

    // Sets default values for best F-score and threshold.
    best_F1_ = 0.0;
    best_epsilon_ = 0.0;
  }

  ~AnomalyDetection() {}

  // Estimates mean and variance of training features.
  int EstimateGaussian(const DataUnlabeled &data_unlabeled);

  // Computes probability of each training example.
  int MultivariateGaussian(const Data &data,int cross_val_flag);
  
  // Find the best threshold for detecting anomalies.
  int SelectThreshold(const Data &data);

  inline arma::vec data_mean() const {
    return data_mean_;
  }

  inline arma::vec data_variance() const {
    return data_variance_;
  }

  inline arma::vec data_probs() const {
    return data_probs_;
  }

  inline arma::vec data_cross_val_probs() const {
    return data_cross_val_probs_;
  }

  inline double best_F1() const {
    return best_F1_;
  }

  inline double best_epsilon() const {
    return best_epsilon_;
  }

  inline int set_data_mean(arma::vec data_mean_arg) {
    data_mean_ = data_mean_arg;

    return 0;
  }

  inline int set_data_variance(arma::vec data_variance_arg) {
    data_variance_ = data_variance_arg;

    return 0;
  }

  inline int set_data_probs(arma::vec data_probs_arg) {
    data_probs_ = data_probs_arg;

    return 0;
  }

  inline int set_data_cross_val_probs(arma::vec data_cross_val_probs_arg) {
    data_cross_val_probs_ = data_cross_val_probs_arg;

    return 0;
  }

  inline int set_best_F1(double best_F1_arg) {
    best_F1_ = best_F1_arg;

    return 0;
  }

  inline int set_best_epsilon(double best_epsilon_arg) {
    best_epsilon_ = best_epsilon_arg;

    return 0;
  }

 private:
  // Mean of training features.
  arma::vec data_mean_;

  // Variance of training features.
  arma::vec data_variance_;

  // Probabilities of training examples.
  arma::vec data_probs_;

  // Probabilities of cross-validation examples.
  arma::vec data_cross_val_probs_;

  // Best F-score.
  double best_F1_;

  // Threshold that yields best F-score.
  double best_epsilon_;

  DISALLOW_COPY_AND_ASSIGN(AnomalyDetection);
};

#endif	// ANOMALY_DETECTION_H_
