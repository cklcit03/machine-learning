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

// LogisticRegression class 1) implements key functions for logistic regression
// and 2) stores relevant parameters.
// RegularizedLogisticRegression class inherits from LogisticRegression class; 
// it relies on a regularization parameter, "lambda".
// MultiClassRegularizedLogisticRegression class inherits from 
// RegularizedLogisticRegression class; it applies to the case where we have
// more than two class labels.

#ifndef LOGISTIC_REGRESSION_H_
#define LOGISTIC_REGRESSION_H_

#include <assert.h>
#include <string>

#include "armadillo"
#include "nlopt.hpp"

#include "data.h"

// Implements key functions for logistic regression and stores relevant 
// parameters.
// Sample usage:
// LogisticRegression log_reg(kIterations,theta_mat,grad_vec,pred_vec);
// const double kInitReturnCode = log_reg.ComputeGradient(application_data);
class LogisticRegression
{
 public:
  // Sets default values for algorithm parameters.
  LogisticRegression() {
    num_iters_ = 1;
    theta_.zeros(3,1);
    gradient_.zeros(3,1);
    predictions_.zeros(3,1);
  }

  // Sets values for algorithm parameters.
  // "num_iters_arg" corresponds to the number of iterations.
  // "theta_arg" corresponds to an initial guess of the weights.
  // "predictions_arg" corresponds to an initial guess of the training label
  // predictions.
  LogisticRegression(int num_iters_arg,arma::mat theta_arg,
    arma::vec gradient_arg,arma::vec predictions_arg) : \
      num_iters_(num_iters_arg),theta_(theta_arg),gradient_(gradient_arg),
      predictions_(predictions_arg) {}

  ~LogisticRegression() {}

  // Computes sigmoid function.
  // Given an argument x, computes the following:
  // 1 / (1 + exp(-x))
  arma::vec ComputeSigmoid(const arma::vec sigmoid_arg);

  // Computes cost function given training data in "data" and current 
  // weights in theta_.
  // Cost function term (for each training example) is: 
  // (-1 / (number of training examples)) * 
  // ((training label) * log(sigmoid(theta_ * training features)) + 
  // (1 - (training label))*log(1-sigmoid(theta_ * training features)))
  double ComputeCost(const std::vector<double> &theta,
    std::vector<double> &grad,const Data &data);

  // Computes gradient given training data in "data" and current
  // weights in theta_.
  // Gradient term (for each training example and each training feature) is:
  // (1 / (number of training examples)) * 
  // (sigmoid(theta_ * training features) - (training label)) * 
  // (training features)
  int ComputeGradient(const Data &data);

  // Computes label predictions given training data in "data" and current
  // weights in theta_.
  // If probability is at least 0.5, assigns example to class "1".
  // If probability is less than 0.5, assigns example to class "0".
  int LabelPrediction(const Data &data);

  inline virtual arma::mat theta() const {
    return theta_;
  }

  inline virtual arma::vec gradient() const {
    return gradient_;
  }

  inline virtual arma::vec predictions() const {
    return predictions_;
  }

  inline virtual int num_iters() const {
    return num_iters_;
  }

  inline virtual int set_theta(arma::mat theta_arg) {
    theta_ = theta_arg;

    return 0;
  }

  inline virtual int set_gradient(arma::vec gradient_arg) {
    gradient_ = gradient_arg;

    return 0;
  }

  inline virtual int set_predictions(arma::vec predictions_arg) {
    predictions_ = predictions_arg;

    return 0;
  }
  
  inline virtual int set_num_iters(int num_iters_arg) {
    num_iters_ = num_iters_arg;

    return 0;
  }

 private:
  // Current weights for logistic regression.
  arma::mat theta_;

  // Current gradient for logistic regression.
  arma::vec gradient_;

  // Current training label predictions.
  arma::vec predictions_;

  int num_iters_;

  DISALLOW_COPY_AND_ASSIGN(LogisticRegression);
};

// Implements key functions for regularized logistic regression and stores 
// relevant parameters.
// Sample usage:
// RegularizedLogisticRegression reg_log_reg(kIterations,theta_mat,grad_vec,pred_vec,kLambda);
// const double kInitReturnCode = reg_log_reg.ComputeGradient(application_data);
class RegularizedLogisticRegression: public LogisticRegression
{
 public:
  // Sets default values for algorithm parameters.
  RegularizedLogisticRegression() : LogisticRegression() {
    lambda_ = 1.0;
  }

  // Sets values for algorithm parameters.
  // Use constructor for LogisticRegression given "num_iters_arg", "theta_arg",
  // "gradient_arg" and "predictions_arg".
  // "lambda_arg" corresponds to the regularization parameter.
  RegularizedLogisticRegression(int num_iters_arg,arma::mat theta_arg,
    arma::vec gradient_arg,arma::vec predictions_arg,double lambda_arg) : \
    LogisticRegression(num_iters_arg,theta_arg,gradient_arg,predictions_arg),\
    lambda_(lambda_arg) {}

  ~RegularizedLogisticRegression() {}

  // Computes cost function given training data in "data", current 
  // weights in theta_ and current regularization parameter lambda_.
  // Cost function term (for each training example) is (w/o regularization): 
  // (-1 / (number of training examples)) * 
  // ((training label) * log(sigmoid(theta_ * training features)) + 
  // (1 - (training label))*log(1-sigmoid(theta_ * training features)))
  // Sums all of these terms to obtain standard logistic regression cost.
  // To this cost, adds following term:
  // (lambda_ / (2 * (number of training examples))) * sum(theta_ * theta_)
  double ComputeCost(const std::vector<double> &opt_param,
    std::vector<double> &grad,const Data &data);

  // Computes gradient given training data in "data", current
  // weights in theta_ and current regularization parameter lambda_.
  // Gradient term (for each training example and each training feature) is
  // (w/o regularization):
  // (1 / (number of training examples)) * 
  // (sigmoid(theta_ * training features) - (training label)) * 
  // (training features)
  // For each training feature, sums all of these terms (over all training 
  // examples) to obtain standard logistic regression gradient.
  // To this gradient, adds following term (except for first dummy feature):
  // (lambda_ / (number of training examples)) * theta_(feature index)
  int ComputeGradient(const Data &data);

  inline double lambda() const {
    return lambda_;
  }

  inline int set_lambda(double lambda_arg) {
    lambda_ = lambda_arg;

    return 0;
  }

 private:
  double lambda_;

  DISALLOW_COPY_AND_ASSIGN(RegularizedLogisticRegression);
};

// Implements key functions for multi-class regularized logistic regression 
// and stores relevant parameters.
// Sample usage:
// MultiClassRegularizedLogisticRegression mul_class_reg_log_reg(kIterations,theta_mat,grad_vec,pred_vec,kLambda,kNumClass);
// const double kInitReturnCode = mul_class_reg_log_reg.ComputeGradient(application_data,class_label);
class MultiClassRegularizedLogisticRegression: public RegularizedLogisticRegression
{
 public:
  // Sets default values for algorithm parameters.
  MultiClassRegularizedLogisticRegression() : RegularizedLogisticRegression() {
    num_class_ = 3;
  }

  // Sets values for algorithm parameters.
  // Use constructor for RegularizedLogisticRegression given "num_iters_arg", 
  // "theta_arg", "gradient_arg", "predictions_arg" and "lambda_arg"
  // "num_class_arg" corresponds to the number of class labels.
  MultiClassRegularizedLogisticRegression(int num_iters_arg,\
    arma::mat theta_arg,arma::vec gradient_arg,arma::vec predictions_arg,\
	double lambda_arg,int num_class_arg) : \
    RegularizedLogisticRegression(num_iters_arg,theta_arg,gradient_arg,\
    predictions_arg,lambda_arg),num_class_(num_class_arg) {}

  ~MultiClassRegularizedLogisticRegression() {}

  // Computes sigmoid function for matrix arguments.
  arma::mat ComputeSigmoid(const arma::mat sigmoid_arg);

  // Computes cost function given training data in "data_multi", current 
  // weights in theta_ and current regularization parameter lambda_.
  // Sets training labels for one-vs-all classification given "class_label".
  // Uses weights in theta_ given "class_label".
  // Cost function term (for each training example) is (w/o regularization): 
  // (-1 / (number of training examples)) * 
  // ((training label) * log(sigmoid(theta_ * training features)) + 
  // (1 - (training label))*log(1-sigmoid(theta_ * training features)))
  // Sums all of these terms to obtain standard logistic regression cost.
  // To this cost, adds following term:
  // (lambda_ / (2 * (number of training examples))) * sum(theta_ * theta_)
  double ComputeCost(const std::vector<double> &opt_param,
    std::vector<double> &grad,const DataMulti &data_multi,
    const int &class_label);

  // Computes gradient given training data in "data_multi", current
  // weights in theta_ and current regularization parameter lambda_.
  // Sets training labels for one-vs-all classification given "class_label".
  // Uses weights in theta_ given "class_label".
  // Gradient term (for each training example and each training feature) is
  // (w/o regularization):
  // (1 / (number of training examples)) * 
  // (sigmoid(theta_ * training features) - (training label)) * 
  // (training features)
  // For each training feature, sums all of these terms (over all training 
  // examples) to obtain standard logistic regression gradient.
  // To this gradient, adds following term (except for first dummy feature):
  // (lambda_ / (number of training examples)) * theta_(feature index)
  int ComputeGradient(const DataMulti &data_multi,const int &class_label);

  // Performs one-versus-all classification given training data in 
  // "data_multi".
  // Loops through class labels and trains a binary classifier for each label.
  int OneVsAll(DataMulti &data_multi);

  // Computes label predictions given training data in "data" and current
  // weights in theta_.
  // For each example, computes sigmoid function for each class.
  // Assigns example to class with highest value of sigmoid function.
  int LabelPrediction(const DataMulti &data_multi);

  inline int num_class() const {
    return num_class_;
  }

  inline int set_num_class(int num_class_arg) {
    num_class_ = num_class_arg;

    return 0;
  }

 private:
  int num_class_;

  DISALLOW_COPY_AND_ASSIGN(MultiClassRegularizedLogisticRegression);
};

// Defines a struct that contains an instance of 
// MultiClassRegularizedLogisticRegression class, an instance of DataMulti 
// class and a class label.  This struct will be passed as "void_data" to 
// ComputeCostWrapper.
struct WrapperStruct {
  MultiClassRegularizedLogisticRegression *mul_class_reg_log_reg;
  DataMulti *data_multi;
  int class_label;
};

// nlopt requires a wrapper function.  A WrapperStruct is contained in
// void_data, and it is unpacked in this wrapper function.
// This wrapper function calls ComputeCost to update "opt_param" and "grad".
double ComputeCostWrapper(const std::vector<double> &opt_param,
  std::vector<double> &grad,void *void_data);

#endif	// LOGISTIC_REGRESSION_H_
