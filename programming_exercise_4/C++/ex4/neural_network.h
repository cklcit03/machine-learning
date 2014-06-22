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

// NeuralNetwork class 1) implements key functions for a neural network
// and 2) stores relevant parameters.

#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <assert.h>
#include <string>

#include "armadillo"
#include "nlopt.hpp"

#include "data.h"

// Implements key functions for a neural network and stores relevant 
// parameters.
// Sample usage:
// NeuralNetwork neu_net(theta_mat,pred_vec);
// const double kInitReturnCode = neu_net.LabelPrediction(application_data);
class NeuralNetwork
{
 public:
  // Sets default values for algorithm parameters.
  NeuralNetwork() {
    predictions_.zeros(3,1);
    num_layers_ = 2;
  }

  // Sets values for algorithm parameters.
  // "theta_arg" corresponds to an initial guess of the weights.
  // "predictions_arg" corresponds to an initial guess of the training label
  // predictions.
  // "num_layers_arg" corresponds to the number of neural network layers (with
  // the exception of the output layer)
  NeuralNetwork(std::vector<arma::mat> theta_arg,arma::vec predictions_arg,\
    int num_layers_arg) : predictions_(predictions_arg),\
    num_layers_(num_layers_arg) {
    theta_ = theta_arg;
	input_layer_size_ = theta_.at(0).n_cols-1;
	hidden_layer_size_ = theta_.at(1).n_cols-1;
	output_layer_size_ = theta_.at(1).n_rows;
  }

  ~NeuralNetwork() {}

  // Computes sigmoid function.
  // Given an argument x, computes the following:
  // 1 / (1 + exp(-x))
  arma::mat ComputeSigmoid(const arma::mat sigmoid_arg);

  // Computes gradient of sigmoid function.
  // Given an argument x, computes the following:
  // exp(-x) / ((1 + exp(-x)) * (1 + exp(-x))
  arma::mat ComputeSigmoidGradient(const arma::mat sigmoid_arg);

  // Performs random initialization of weights in theta_.
  int RandInitializeWeights();

  // Computes cost function given training data in "data_multi", current 
  // weights in theta_ and current regularization parameter lambda_.
  // For each training example, sets training label for actual class equal to 
  // 1; sets training labels for all other classes equal to 0.
  // Cost function term (for each training example and each class) is (w/o 
  // regularization): 
  // (-1 / (number of training examples)) * 
  // ((training label) * log(activation of output unit for this class) + 
  // (1 - (training label))*log(1-activation of output unit for this class))
  // Sums all of these terms to obtain standard neural network cost.
  // To this cost, adds following term:
  // (lambda_ / (2 * (number of training examples))) * (squared Frobenius norm of theta_)
  double ComputeCost(const std::vector<double> &opt_param,
    std::vector<double> &grad,const DataMulti &data_multi);

  // Computes gradient given training data in "data_multi".
  // Uses backpropagation to compute this quantity.
  // For each example, performs following steps.
  // Step 1: runs feedforward section of neural network to compute activation
  // of hidden and output layers.
  // Step 2: computes "error term" for output layer.
  // Step 3: computes "error term" for hidden layer.
  // Step 4: accumulates gradient term using above-mentioned "error terms".
  // After iterating over all examples, perform following steps. 
  // Step 5: divides accumulated gradient term by number of training examples
  // to obtain unregularized gradient.
  // Step 6: scales weights in theta_ by 
  // (lambda_/(number of training examples)) and adds this term to
  // unregularized gradient to obtain regularized gradient.
  int ComputeGradient(const DataMulti &data_multi);

  // Computes label predictions given training data in "data_multi" and current
  // weights in theta_.
  // For each example, computes sigmoid function for each class.
  // Assigns example to class with highest value of sigmoid function.
  int LabelPrediction(const DataMulti &data_multi);

  inline virtual std::vector<arma::mat> theta() const {
    return theta_;
  }

  inline virtual arma::vec gradient() const {
    return gradient_;
  }

  inline virtual arma::vec predictions() const {
    return predictions_;
  }

  inline virtual int input_layer_size() const {
    return input_layer_size_;
  }

  inline virtual int hidden_layer_size() const {
    return hidden_layer_size_;
  }

  inline virtual int output_layer_size() const {
    return output_layer_size_;
  }

  inline double lambda() const {
    return lambda_;
  }

  inline virtual int num_layers() const {
    return num_layers_;
  }

  inline virtual int set_theta(std::vector<arma::mat> theta_arg) {
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

  inline virtual int set_input_layer_size(int input_layer_size_arg) {
    input_layer_size_ = input_layer_size_arg;

    return 0;
  }

  inline virtual int set_hidden_layer_size(int hidden_layer_size_arg) {
    hidden_layer_size_ = hidden_layer_size_arg;

    return 0;
  }

  inline virtual int set_output_layer_size(int output_layer_size_arg) {
    output_layer_size_ = output_layer_size_arg;

    return 0;
  }

  inline int set_lambda(double lambda_arg) {
    lambda_ = lambda_arg;

    return 0;
  }
  
  inline virtual int set_num_layers(int num_layers_arg) {
    num_layers_ = num_layers_arg;

    return 0;
  }

 private:
  // Current weights.
  std::vector<arma::mat> theta_;

  // Current gradient.
  arma::vec gradient_;

  // Current training label predictions.
  arma::vec predictions_;

  // Number of units in input layer.
  int input_layer_size_;

  // Number of units in hidden layer.
  int hidden_layer_size_;

  // Number of units in output layer.
  int output_layer_size_;

  double lambda_;
  
  int num_layers_;

  DISALLOW_COPY_AND_ASSIGN(NeuralNetwork);
};

// Defines a struct that contains an instance of NeuralNetwork 
// class and an instance of DataMulti class.  This struct will be passed as 
// "void_data" to ComputeCostWrapper.
struct WrapperStruct {
  NeuralNetwork *neu_net;
  DataMulti *data_multi;
};

// nlopt requires a wrapper function.  A WrapperStruct is contained in
// void_data, and it is unpacked in this wrapper function.
// This wrapper function calls ComputeCost to update "opt_param" and "grad".
double ComputeCostWrapper(const std::vector<double> &opt_param,
  std::vector<double> &grad,void *void_data);

#endif	// NEURAL_NETWORK_H_
