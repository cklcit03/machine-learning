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

// Defines function that generates values for a learning curve.

#include "learning_curve.h"

// Uses nlopt functionality for training.
int LearningCurve(DataDebug &data_debug,LinearRegression &lin_reg,\
  double *error_train,double *error_val,int use_poly) {
  const int kNumTrainEx = data_debug.num_train_ex();
  for(int ex_index=0; ex_index<kNumTrainEx; ex_index++)
  {

    // Determine if we need to use mapped features.
    if (use_poly == 0) {
      data_debug.set_features(data_debug.training_features().rows(0,ex_index));
    }
    else {
      data_debug.set_features(data_debug.features_normalized().rows(0,ex_index));
    }
    data_debug.set_labels(data_debug.training_labels().rows(0,ex_index));
    const int kFeatures = data_debug.features().n_cols;
    std::vector<double> theta_stack_vec(kFeatures,1.0);
    std::vector<double> grad_vec(kFeatures,0.0);
    lin_reg.set_lambda(1.0);
    lin_reg.Train(data_debug);
    for(unsigned int f_index=0; f_index<(unsigned)kFeatures; f_index++)
    {
      theta_stack_vec.at(f_index) = \
        arma::as_scalar(lin_reg.theta().row(f_index));
    }
    lin_reg.set_lambda(0.0);
    error_train[ex_index] = \
      lin_reg.ComputeCost(theta_stack_vec,grad_vec,data_debug);

    // Determine if we need to use mapped features.
    if (use_poly == 0) {
      data_debug.set_features(data_debug.validation_features());
    }
    else {
      data_debug.set_features(data_debug.validation_features_normalized());
    }
    data_debug.set_labels(data_debug.validation_labels());
    error_val[ex_index] = \
      lin_reg.ComputeCost(theta_stack_vec,grad_vec,data_debug);
  }

  return 0;
}
