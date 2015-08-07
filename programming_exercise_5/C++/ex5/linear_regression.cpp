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

// LinearRegression class 1) implements key functions for linear regression
// and 2) stores relevant parameters.

#include "linear_regression.h"

// Arguments "opt_param" and "grad" are updated by nlopt.
// "opt_param" corresponds to the optimization parameters.
// "grad" corresponds to the gradient.
double LinearRegression::ComputeCost(const std::vector<double> &opt_param,
  std::vector<double> &grad,const DataDebug &data_debug) {
  const int kNumTrainEx = data_debug.features().n_rows;
  assert(kNumTrainEx >= 1);
  const int kNumFeatures = data_debug.features().n_cols;
  assert(kNumFeatures >= 1);
  arma::vec nlopt_theta = arma::randu<arma::vec>(kNumFeatures,1);

  // Uses current value of "theta" from nlopt.
  for(int feature_index=0; feature_index<kNumFeatures; feature_index++)
  {
    nlopt_theta(feature_index) = opt_param[feature_index];
  }
  set_theta(nlopt_theta);

  // Computes cost function given current value of "theta".
  const arma::mat kTrainingFeatures = data_debug.features();
  const arma::vec kTrainingLabels = data_debug.labels();
  const arma::vec kDiffVec = kTrainingFeatures*theta_-kTrainingLabels;
  const arma::vec kDiffVecSq = kDiffVec % kDiffVec;
  const double kJTheta = arma::as_scalar(sum(kDiffVecSq))/(2.0*kNumTrainEx);

  // Adds regularization term.
  const arma::vec kThetaSquared = theta_%theta_;
  const double kRegTerm = (lambda_/(2*kNumTrainEx))*\
    (sum(kThetaSquared)-kThetaSquared(0));
  const double kJThetaReg = kJTheta+kRegTerm;

  // Updates "grad" for nlopt.
  const int kReturnCode = this->ComputeGradient(data_debug);
  const arma::vec kCurrentGradient = gradient();
  for(int feature_index=0; feature_index<kNumFeatures; feature_index++)
  {
    grad[feature_index] = kCurrentGradient(feature_index);
  }

  return kJThetaReg;
}

// The number of training examples should always be a positive integer.
int LinearRegression::ComputeGradient(const DataDebug &data_debug) {
  const int kNumTrainEx = data_debug.features().n_rows;
  assert(kNumTrainEx >= 1);
  const int kNumFeatures = data_debug.features().n_cols;
  assert(kNumFeatures >= 1);

  const arma::mat kTrainingFeatures = data_debug.features();
  const arma::vec kTrainingLabels = data_debug.labels();
  const arma::vec kCurrentTheta = theta();

  arma::vec gradient_array = arma::zeros<arma::vec>(kNumFeatures);
  arma::vec gradient_array_reg = arma::zeros<arma::vec>(kNumFeatures);

  for(int feature_index=0; feature_index<kNumFeatures; feature_index++)
  {
    const arma::vec gradient_term = \
      (kTrainingFeatures*kCurrentTheta-kTrainingLabels) % \
      kTrainingFeatures.col(feature_index);
    gradient_array(feature_index) = sum(gradient_term)/kNumTrainEx;
    gradient_array_reg(feature_index) = gradient_array(feature_index)+\
      (lambda_/kNumTrainEx)*(kCurrentTheta(feature_index));
  }
  gradient_array_reg(0) -= (lambda_/kNumTrainEx)*(kCurrentTheta(0));
  set_gradient(gradient_array_reg);

  return 0;
}

// Uses nlopt functionality for training.
int LinearRegression::Train(DataDebug &data_debug) {
  WrapperStruct wrap_struct;
  wrap_struct.lin_reg = this;
  wrap_struct.data_debug = &data_debug;
  nlopt::opt opt(nlopt::LD_LBFGS,data_debug.features().n_cols);
  opt.set_min_objective(ComputeCostWrapper,&wrap_struct);
  opt.set_ftol_abs(1e-6);
  std::vector<double> nlopt_theta(data_debug.features().n_cols,1.0);
  double min_cost = 0.0;
  nlopt::result nlopt_result = opt.optimize(nlopt_theta,min_cost);

  return 0;
}

// Unpacks WrapperStruct to obtain instances of LinearRegression 
// and DataDebug.
double ComputeCostWrapper(const std::vector<double> &opt_param,
    std::vector<double> &grad,void *void_data) {
  WrapperStruct *wrap_struct = static_cast<WrapperStruct *>(void_data);
  LinearRegression *lin_reg = wrap_struct->lin_reg;
  DataDebug *data_debug = wrap_struct->data_debug;

  return lin_reg->ComputeCost(opt_param,grad,*data_debug);
}
