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

// NeuralNetwork class functions implement modules for the purpose of training
// a binary (or multi-class) classifier.

#include "neural_network.h"

// This function is applied to matrices.
arma::mat NeuralNetwork::ComputeSigmoid(const arma::mat sigmoid_arg) {
  const arma::mat kSigmoid = 1/(1+arma::exp(-sigmoid_arg));

  return kSigmoid;
}

// The number of layers should be no less than 2.
// The number of training examples should always be a positive integer.
int NeuralNetwork::LabelPrediction (const DataMulti &data_multi) {
  assert(num_layers_ > 1);
  const int kNumTrainEx = data_multi.num_train_ex();
  assert(kNumTrainEx > 0);
  arma::mat layer_activation_in = data_multi.training_features();
  arma::mat sigmoid_arg = layer_activation_in*theta_.at(0).t();
  arma::mat layer_activation_out = ComputeSigmoid(sigmoid_arg);
  for(int layer_index=1; layer_index<num_layers_; layer_index++)
  {
    const arma::mat kOnesMat = arma::ones(layer_activation_out.n_rows,1);
    layer_activation_in = arma::join_horiz(kOnesMat,layer_activation_out);
    sigmoid_arg = layer_activation_in*theta_.at(layer_index).t();
    layer_activation_out = ComputeSigmoid(sigmoid_arg);
  }
  arma::vec current_predictions = predictions();
  for(int example_index=0; example_index<kNumTrainEx; example_index++)
  {
    int curr_max_index = 0;
    double curr_max_val = layer_activation_out.row(example_index)(0);
    for(int label_index=1; label_index<data_multi.num_labels(); label_index++)
    {
      if (layer_activation_out.row(example_index)(label_index) > curr_max_val)
      {
        curr_max_val = layer_activation_out.row(example_index)(label_index);
        curr_max_index = label_index;
      }
    }
    current_predictions(example_index) = curr_max_index+1;
  }
  set_predictions(current_predictions);

  return 0;
}