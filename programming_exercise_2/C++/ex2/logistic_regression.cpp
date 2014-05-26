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

// LogisticRegression class functions implement modules that are then passed
// to an unconstrained minimization algorithm.

#include "logistic_regression.h"

// This function is applied to vectors.
arma::vec LogisticRegression::ComputeSigmoid(const arma::vec sigmoid_arg) {
  const arma::vec kSigmoid = 1/(1+arma::exp(-sigmoid_arg));

  return kSigmoid;
}

// Arguments "theta" and "grad" are updated by nlopt.
// "theta" corresponds to the optimization parameters.
// "grad" corresponds to the gradient.
double LogisticRegression::ComputeCost(const std::vector<double> &theta,
	std::vector<double> &grad,const Data &data) {
  const int kNumFeatures = data.num_features();
  arma::vec nlopt_theta = arma::randu<arma::vec>(kNumFeatures+1,1);

  // Use current value of "theta" from nlopt.
  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
    nlopt_theta(feature_index) = theta[feature_index];
  }
  set_theta(nlopt_theta);
  const arma::mat kTrainingFeatures = data.training_features();
  const arma::vec kTrainingLabels = data.training_labels();
  const arma::vec kSigmoidArg = kTrainingFeatures*theta_;
  const arma::vec kSigmoidVal = ComputeSigmoid(kSigmoidArg);
  const int kNumTrainEx = data.num_train_ex();
  assert(kNumTrainEx > 0);
  const arma::vec kCostFuncVal = (-1)*(kTrainingLabels%arma::log(kSigmoidVal)+\
    (1-kTrainingLabels)%arma::log(1-kSigmoidVal));
  const double kJTheta = sum(kCostFuncVal)/kNumTrainEx;

  // Update "grad" for nlopt.
  const int kReturnCode = this->ComputeGradient(data);
  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
	grad[feature_index] = gradient_(feature_index);
  }

  return kJTheta;
}

// The number of training examples should always be a positive integer.
int LogisticRegression::ComputeGradient(const Data &data) {
  const int kNumTrainEx = data.num_train_ex();
  assert(kNumTrainEx > 0);
  const arma::mat kTrainingFeatures = data.training_features();
  const int kNumFeatures = data.num_features();
  assert(kNumFeatures > 0);
  const arma::vec kTrainingLabels = data.training_labels();

  const arma::vec kSigmoidArg = kTrainingFeatures*theta_;
  const arma::vec kSigmoidVal = ComputeSigmoid(kSigmoidArg);

  arma::vec gradient_array = arma::zeros<arma::vec>(kNumFeatures+1);

  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
	const arma::vec gradient_term = \
      (kSigmoidVal-kTrainingLabels) % kTrainingFeatures.col(feature_index);
	gradient_array(feature_index) = sum(gradient_term)/kNumTrainEx;
  }
  set_gradient(gradient_array);

  return 0;
}

// The number of training examples should always be a positive integer.
int LogisticRegression::LabelPrediction(const Data &data) {
  const int kNumTrainEx = data.num_train_ex();
  assert(kNumTrainEx > 0);
  const arma::mat kTrainingFeatures = data.training_features();
  const arma::vec kSigmoidArg = kTrainingFeatures*theta_;
  const arma::vec kSigmoidVal = ComputeSigmoid(kSigmoidArg);
  for(int example_index=0; example_index<kNumTrainEx; example_index++)
  {
    predictions_(example_index) = (kSigmoidVal(example_index) >= 0.5) ? 1 : 0;
  }

  return 0;
}

// Unpacks WrapperStruct to obtain instances of LogisticRegression and Data.
double ComputeCostWrapper(const std::vector<double> &theta,
	std::vector<double> &grad,void *void_data) {
  WrapperStruct *wrap_struct = static_cast<WrapperStruct *>(void_data);
  LogisticRegression *log_res = wrap_struct->log_res;
  Data *data = wrap_struct->data;

  return log_res->ComputeCost(theta,grad,*data);
}
