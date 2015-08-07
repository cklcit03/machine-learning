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

// Defines function that generates values for a cross-validation curve.

#include "validation_curve.h"

// Uses nlopt functionality for training.
int ValidationCurve(DataDebug &data_debug,LinearRegression &lin_reg,\
  double *error_train,double *error_val,arma::rowvec lambda_vec) {
  for(int lambda_index=0; lambda_index<10; lambda_index++)
  {
    data_debug.set_features(data_debug.features_normalized());
    const int kFeatures = data_debug.features().n_cols;
    assert(kFeatures >= 1);
    data_debug.set_labels(data_debug.training_labels());
    std::vector<double> theta_stack_vec(kFeatures,1.0);
    std::vector<double> grad_vec(kFeatures,0.0);
    lin_reg.set_lambda(lambda_vec(lambda_index));
    lin_reg.Train(data_debug);
    for(unsigned int f_index=0; f_index<(unsigned)kFeatures; f_index++)
    {
      theta_stack_vec.at(f_index) = \
        arma::as_scalar(lin_reg.theta().row(f_index));
    }
    lin_reg.set_lambda(0.0);
    error_train[lambda_index] = \
      lin_reg.ComputeCost(theta_stack_vec,grad_vec,data_debug);
    data_debug.set_features(data_debug.validation_features_normalized());
    data_debug.set_labels(data_debug.validation_labels());
    error_val[lambda_index] = \
      lin_reg.ComputeCost(theta_stack_vec,grad_vec,data_debug);
  }

  return 0;
}
