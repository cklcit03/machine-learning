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

// This function is applied to matrices.
arma::mat NeuralNetwork::ComputeSigmoidGradient(const arma::mat sigmoid_arg) {
  const arma::mat kSigmoidGradient = \
    (arma::exp(-sigmoid_arg))/(arma::square(1+arma::exp(-sigmoid_arg)));

  return kSigmoidGradient;
}

// This function initializes theta_ for nlopt.
int NeuralNetwork::RandInitializeWeights() {
  double epsilon_init = 0.12;
  assert(num_layers_ >= 0);
  for(int layer_index=0; layer_index<num_layers_; layer_index++)
  {
    arma::mat layer_theta = theta_.at(layer_index);
    int num_in_units = layer_theta.n_cols;
    int num_out_units = layer_theta.n_rows;
    arma::mat layer_theta_rand = \
      2*epsilon_init*arma::randu<arma::mat>(num_out_units,num_in_units)-\
      epsilon_init*arma::ones<arma::mat>(num_out_units,num_in_units);
	theta_.at(layer_index) = layer_theta_rand;
  }

  return(0);
}

// Arguments "opt_param" and "grad" are updated by nlopt.
// "opt_param" corresponds to the optimization parameters.
// "grad" corresponds to the gradient.
double NeuralNetwork::ComputeCost(const std::vector<double> &opt_param,
  std::vector<double> &grad,const DataMulti &data_multi) {
  const int kNumFeatures = hidden_layer_size_*(input_layer_size_+1)+\
    output_layer_size_*(hidden_layer_size_+1);
  arma::vec nlopt_theta = arma::randu<arma::vec>(kNumFeatures,1);

  // Use current value of "theta" from nlopt.
  for(int feature_index=0; feature_index<kNumFeatures; feature_index++)
  {
    nlopt_theta(feature_index) = opt_param[feature_index];
  }
  
  // Extract Theta1 and Theta2 from nlopt_theta.
  const arma::vec kNLoptTheta1 = \
    nlopt_theta.rows(0,hidden_layer_size_*(input_layer_size_+1)-1);
  const arma::vec kNLoptTheta2 = \
    nlopt_theta.rows(hidden_layer_size_*(input_layer_size_+1),\
      hidden_layer_size_*(input_layer_size_+1)+\
        output_layer_size_*(hidden_layer_size_+1)-1);

  // Reshape Theta1 and Theta2 as matrices.
  const arma::mat kNLoptTheta1Mat = \
    arma::reshape(kNLoptTheta1,hidden_layer_size_,input_layer_size_+1);
  const arma::mat kNLoptTheta2Mat = \
    arma::reshape(kNLoptTheta2,output_layer_size_,hidden_layer_size_+1);
  std::vector<arma::mat> current_theta;
  current_theta.push_back(kNLoptTheta1Mat);
  current_theta.push_back(kNLoptTheta2Mat);
  set_theta(current_theta);

  // Run feedforward propagation.
  const int kNumTrainEx = data_multi.num_train_ex();
  assert(kNumTrainEx > 0);
  arma::mat layer_activation_in = data_multi.training_features();
  arma::mat sigmoid_arg = layer_activation_in*theta_.at(0).t();
  arma::mat layer_activation_out = ComputeSigmoid(sigmoid_arg);
  assert(num_layers_ > 1);
  for(int layer_index=1; layer_index<num_layers_; layer_index++)
  {
    const arma::mat kOnesMat = arma::ones(layer_activation_out.n_rows,1);
    layer_activation_in = arma::join_horiz(kOnesMat,layer_activation_out);
    sigmoid_arg = layer_activation_in*theta_.at(layer_index).t();
    layer_activation_out = ComputeSigmoid(sigmoid_arg);
  }

  // Set up matrix of (multi-class) training labels.
  arma::mat label_mat = arma::randu<arma::mat>(kNumTrainEx,output_layer_size_);
  label_mat.zeros(kNumTrainEx,output_layer_size_);
  for(int example_index=0; example_index<kNumTrainEx; example_index++)
  {
    int col_index = \
      (int)arma::as_scalar(data_multi.training_labels()(example_index));
    label_mat(example_index,col_index-1) = 1.0;
  }

  // Compute regularized neural network cost.
  const arma::mat kCostFuncVal = \
    (-1)*(label_mat%arma::log(layer_activation_out)+\
    (1-label_mat)%arma::log(1-layer_activation_out));
  const double kJTheta = accu(kCostFuncVal)/kNumTrainEx;
  const arma::mat kNLoptTheta1MatSquared = kNLoptTheta1Mat%kNLoptTheta1Mat;
  const arma::mat kNLoptTheta2MatSquared = kNLoptTheta2Mat%kNLoptTheta2Mat;
  const arma::mat kNLoptTheta1MatSquaredTrans = kNLoptTheta1MatSquared.t();
  const arma::mat kNLoptTheta2MatSquaredTrans = kNLoptTheta2MatSquared.t();
  const double kRegTerm = \
    (lambda()/(2*kNumTrainEx))*(accu(kNLoptTheta1MatSquaredTrans)+\
      accu(kNLoptTheta2MatSquaredTrans)-\
	  accu(kNLoptTheta1MatSquaredTrans.row(0))-\
      accu(kNLoptTheta2MatSquaredTrans.row(0)));
  const double kJThetaReg = kJTheta+kRegTerm;

  // Update "grad" for nlopt.
  const int kReturnCode = this->ComputeGradient(data_multi);
  const arma::vec kCurrentGradient = gradient();
  for(int feature_index=0; feature_index<kNumFeatures; feature_index++)
  {
    grad[feature_index] = kCurrentGradient(feature_index);
  }
  
  printf("kJThetaReg = %.6f\n",kJThetaReg);

  return kJThetaReg;
}

// The number of training examples should always be a positive integer.
int NeuralNetwork::ComputeGradient(const DataMulti &data_multi) {
  const int kNumTrainEx = data_multi.num_train_ex();
  assert(kNumTrainEx > 0);
  arma::mat accum_hidden_layer_grad = \
    arma::zeros<arma::mat>(theta_.at(0).n_rows,\
      data_multi.training_features().n_cols);
  arma::mat accum_output_layer_grad = \
    arma::zeros<arma::mat>(theta_.at(1).n_rows,\
      theta_.at(0).n_rows+1);

  // Iterate over the training examples.
  for(int example_index=0; example_index<kNumTrainEx; example_index++)
  {

    // Perform step 1.
	arma::rowvec example_features = \
      data_multi.training_features().row(example_index);
    arma::mat sigmoid_arg = example_features*theta_.at(0).t();
    arma::mat hidden_layer_activation = ComputeSigmoid(sigmoid_arg);
    const arma::mat kOnesMat = arma::ones(hidden_layer_activation.n_rows,1);
    arma::mat hidden_layer_activation_mod = \
      arma::join_horiz(kOnesMat,hidden_layer_activation);
    sigmoid_arg = hidden_layer_activation_mod*theta_.at(1).t();
    arma::mat output_layer_activation = ComputeSigmoid(sigmoid_arg);

    // Perform step 2.
	arma::rowvec training_labels = \
      arma::zeros<arma::rowvec>(1,output_layer_size_);
	int column_index = \
      (int)as_scalar(data_multi.training_labels().row(example_index));
    training_labels(column_index-1) = 1;
    arma::colvec output_layer_error = \
      (output_layer_activation-training_labels).t();

    // Perform step 3.
    arma::colvec hidden_layer_error_term = theta_.at(1).t()*output_layer_error;
    arma::colvec hidden_layer_error = \
      hidden_layer_error_term.rows(1,hidden_layer_size_) % \
	  ComputeSigmoidGradient((example_features*theta_.at(0).t()).t());

    // Perform step 4.
    accum_hidden_layer_grad = \
      accum_hidden_layer_grad+hidden_layer_error*example_features;
    accum_output_layer_grad = \
      accum_output_layer_grad+output_layer_error*\
      arma::join_horiz(arma::ones<arma::mat>(1,1),hidden_layer_activation);
  }

  // Perform step 5 (without regularization).
  arma::mat hidden_layer_grad = (1.0/kNumTrainEx)*accum_hidden_layer_grad;
  arma::mat output_layer_grad = (1.0/kNumTrainEx)*accum_output_layer_grad;

  // Perform step 5 (with regularization).
  hidden_layer_grad.cols(1,theta_.at(0).n_cols-1) = \
    hidden_layer_grad.cols(1,theta_.at(0).n_cols-1)+\
    (lambda_/kNumTrainEx)*theta_.at(0).cols(1,theta_.at(0).n_cols-1);
  output_layer_grad.cols(1,theta_.at(1).n_cols-1) = \
    output_layer_grad.cols(1,theta_.at(1).n_cols-1)+\
    (lambda_/kNumTrainEx)*theta_.at(1).cols(1,theta_.at(1).n_cols-1);

  // Set gradient.
  arma::vec hidden_layer_grad_stack = arma::vectorise(hidden_layer_grad);
  arma::vec output_layer_grad_stack = arma::vectorise(output_layer_grad);
  arma::vec gradient_stack = \
    arma::join_vert(hidden_layer_grad_stack,output_layer_grad_stack);
  set_gradient(gradient_stack);

  return 0;
}

// Unpacks WrapperStruct to obtain instances of NeuralNetwork 
// and DataMulti.
double ComputeCostWrapper(const std::vector<double> &opt_param,
    std::vector<double> &grad,void *void_data) {
  WrapperStruct *wrap_struct = static_cast<WrapperStruct *>(void_data);
  NeuralNetwork *neu_net = wrap_struct->neu_net;
  DataMulti *data_multi = wrap_struct->data_multi;

  return neu_net->ComputeCost(opt_param,grad,*data_multi);
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