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

// GradientDescent class 1) implements gradient descent algorithm and 2) stores
// relevant parameters.

#ifndef GRADIENT_DESCENT_H_
#define GRADIENT_DESCENT_H_

#include <assert.h>
#include <string>

#include "armadillo"

#include "data.h"

// Implements gradient descent algorithm and stores relevant parameters.
// Sample usage:
// GradientDescent grad_des(kAlpha,kIterations,theta_vec);
// const int kReturnCode = grad_des.RunGradientDescent(training_data);
class GradientDescent
{
 public:
  // Sets default values for algorithm parameters.
  GradientDescent() {
    alpha_ = 1.0;
    num_iters_ = 1;
    theta_.zeros(2,1);
  }

  // Sets values for algorithm parameters.
  // "alpha_arg" corresponds to the step size.
  // "num_iters_arg" corresponds to the number of iterations.
  // "theta_arg" corresponds to an initial guess of the weights.
  GradientDescent(double alpha_arg,int num_iters_arg,arma::vec theta_arg) : \
      alpha_(alpha_arg),num_iters_(num_iters_arg),theta_(theta_arg) {}

  ~GradientDescent() {}

  // Computes squared error given training data in "data" and current 
  // weights in theta_.
  // Error term before squaring is: 
  // (training features) * (current weights) - (training labels)
  double ComputeCost(const DataNormalized &data);

  // Runs gradient descent algorithm given training data in "data" and 
  // current algorithm parameters.
  // Calls ComputeCost function at each iteration.
  int RunGradientDescent(const DataNormalized &data);

  inline arma::vec theta() const {
    return theta_;
  }

  inline double alpha() const {
    return alpha_;
  }

  inline int num_iters() const {
    return num_iters_;
  }

  inline int set_theta(arma::vec theta_arg) {
    theta_ = theta_arg;

    return 0;
  }

 private:
  // Current weights for gradient descent.
  arma::vec theta_;

  // Step size for gradient descent.
  double alpha_;

  int num_iters_;

  DISALLOW_COPY_AND_ASSIGN(GradientDescent);
};

#endif	// GRADIENT_DESCENT_H_
