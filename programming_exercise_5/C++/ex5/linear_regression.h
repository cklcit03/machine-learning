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

#ifndef LINEAR_REGRESSION_H_
#define LINEAR_REGRESSION_H_

#include <assert.h>
#include <string>

#include "armadillo"
#include "nlopt.hpp"

#include "data.h"

// Implements key functions for linear regression and stores relevant 
// parameters.
// Sample usage:
// LinearRegression lin_reg(theta_vec,grad_vec);
// const double kInitReturnCode = lin_reg.ComputeGradient(application_data);
class LinearRegression
{
 public:
  // Sets default values for algorithm parameters.
  LinearRegression() {
    theta_.zeros(3,1);
    gradient_.zeros(3,1);
	lambda_ = 0.0;
  }

  // Sets values for algorithm parameters.
  // "theta_arg" corresponds to an initial guess of the weights.
  // "gradient_arg" corresponds to an initial guess of the gradient.
  // "lambda_arg" corresponds to the regularization parameter
  LinearRegression(arma::vec theta_arg,arma::vec gradient_arg,\
    double lambda_arg) : theta_(theta_arg),gradient_(gradient_arg),\
    lambda_(lambda_arg) {}

  ~LinearRegression() {}

  // Computes cost function given training data in "data_debug", current 
  // weights in theta_ and current regularization parameter lambda_.
  // Cost function term (for each training example) is (w/o regularization): 
  // (1 / (2 * (number of training examples))) * 
  // (theta_ * (training features) - (training label))^2
  // Sums all of these terms to obtain standard logistic regression cost.
  // To this cost, adds following term:
  // (lambda_ / (2 * (number of training examples))) * sum(theta_ * theta_)
  double ComputeCost(const std::vector<double> &opt_param,
    std::vector<double> &grad,const DataDebug &data_debug);

  // Computes gradient given training data in "data_debug", current
  // weights in theta_ and current regularization parameter lambda_.
  // Gradient term (for each training example and each training feature) is
  // (w/o regularization):
  // (1 / (number of training examples)) * 
  // (theta_ * (training features) - (training label)) * 
  // (training features)
  // For each training feature, sums all of these terms (over all training 
  // examples) to obtain standard linear regression gradient.
  // To this gradient, adds following term (except for first dummy feature):
  // (lambda_ / (number of training examples)) * theta_(feature index)
  int ComputeGradient(const DataDebug &data_debug);

  // Performs training given training data in "data_debug".
  int Train(DataDebug &data_debug);
  
  inline virtual arma::vec theta() const {
    return theta_;
  }

  inline virtual arma::vec gradient() const {
    return gradient_;
  }

  inline virtual double lambda() const {
    return lambda_;
  }

  inline virtual int set_theta(arma::vec theta_arg) {
    theta_ = theta_arg;

    return 0;
  }

  inline virtual int set_gradient(arma::vec gradient_arg) {
    gradient_ = gradient_arg;

    return 0;
  }

  inline virtual int set_lambda(double lambda_arg) {
    lambda_ = lambda_arg;

    return 0;
  }

 private:
  // Current weights.
  arma::vec theta_;

  // Current gradient.
  arma::vec gradient_;

  double lambda_;

  DISALLOW_COPY_AND_ASSIGN(LinearRegression);
};

// Defines a struct that contains an instance of LinearRegression 
// class and an instance of DataDebug class.  This struct will be passed as 
// "void_data" to ComputeCostWrapper.
struct WrapperStruct {
  LinearRegression *lin_reg;
  DataDebug *data_debug;
};

// nlopt requires a wrapper function.  A WrapperStruct is contained in
// void_data, and it is unpacked in this wrapper function.
// This wrapper function calls ComputeCost to update "opt_param" and "grad".
double ComputeCostWrapper(const std::vector<double> &opt_param,
  std::vector<double> &grad,void *void_data);

#endif	// LINEAR_REGRESSION_H_
