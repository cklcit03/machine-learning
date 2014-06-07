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
  }

  ~NeuralNetwork() {}

  // Computes sigmoid function.
  // Given an argument x, computes the following:
  // 1 / (1 + exp(-x))
  arma::mat ComputeSigmoid(const arma::mat sigmoid_arg);

  // Computes label predictions given training data in "data" and current
  // weights in theta_.
  // For each example, computes sigmoid function for each class.
  // Assigns example to class with highest value of sigmoid function.
  int LabelPrediction(const DataMulti &data_multi);

  inline virtual std::vector<arma::mat> theta() const {
    return theta_;
  }

  inline virtual arma::vec predictions() const {
    return predictions_;
  }

  inline virtual int num_layers() const {
    return num_layers_;
  }

  inline virtual int set_theta(std::vector<arma::mat> theta_arg) {
    theta_ = theta_arg;

    return 0;
  }

  inline virtual int set_predictions(arma::vec predictions_arg) {
    predictions_ = predictions_arg;

    return 0;
  }
  
  inline virtual int set_num_layers(int num_layers_arg) {
    num_layers_ = num_layers_arg;

    return 0;
  }

 private:
  // Current weights for logistic regression.
  std::vector<arma::mat> theta_;

  // Current training label predictions.
  arma::vec predictions_;

  int num_layers_;

  DISALLOW_COPY_AND_ASSIGN(NeuralNetwork);
};

#endif	// NEURAL_NETWORK_H_
