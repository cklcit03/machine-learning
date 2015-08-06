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

// Machine Learning
// Programming Exercise 4: Multi-class Neural Networks
// Problem: Predict label for a handwritten digit given data for pixel values 
// of various handwritten digits

#include "neural_network.h"

int main(void) {
  printf("Loading Data ...\n");
  const std::string kDataFileName = "../../digitData.txt";
  const int kNumLabels = 10;
  DataMulti digit_data(kDataFileName,kNumLabels);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Loads two files that contain parameters trained by a neural network.
  printf("Loading Saved Neural Network Parameters ...\n");
  const std::string kTheta1FileName = "../../Theta1.txt";
  arma::mat theta1;
  theta1.load(kTheta1FileName,arma::csv_ascii);
  const std::string kTheta2FileName = "../../Theta2.txt";
  arma::mat theta2;
  theta2.load(kTheta2FileName,arma::csv_ascii);
  std::vector<arma::mat> theta_vec;
  theta_vec.push_back(theta1);
  theta_vec.push_back(theta2);
  const int kNumTrainEx = digit_data.num_train_ex();
  arma::vec predictions_vec = arma::randu<arma::vec>(kNumTrainEx,1);
  const int kNumLayers = 2;
  NeuralNetwork neu_net(theta_vec,predictions_vec,kNumLayers);
  arma::vec theta1_stack = arma::vectorise(theta1);
  arma::vec theta2_stack = arma::vectorise(theta2);
  arma::vec theta_stack = arma::join_vert(theta1_stack,theta2_stack);
  std::vector<double> theta_stack_vec;
  for(unsigned int unit_index=0; unit_index<theta_stack.n_rows; unit_index++)
  {
    theta_stack_vec.push_back(as_scalar(theta_stack.row(unit_index)));
  }

  // Runs feedforward section of neural network.
  printf("Feedforward Using Neural Network ...\n");
  double lambda = 0.0;
  neu_net.set_lambda(lambda);
  std::vector<double> grad_vec(theta_stack.n_rows,0.0);
  double neural_network_cost = \
    neu_net.ComputeCost(theta_stack_vec,grad_vec,digit_data);
  printf("Cost at parameters (loaded from Theta1.txt and Theta2.txt): %.6f\n",neural_network_cost);
  printf("(this value should be about 0.287629)\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Runs feedforward section of neural network with regularization.
  printf("Checking Cost Function (w/ Regularization) ...\n");
  lambda = 1.0;
  neu_net.set_lambda(lambda);
  neural_network_cost = \
    neu_net.ComputeCost(theta_stack_vec,grad_vec,digit_data);
  printf("Cost at parameters (loaded from Theta1.txt and Theta2.txt): %.6f\n",neural_network_cost);
  printf("(this value should be about 0.383770)\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Computes gradient for sigmoid function.
  printf("Evaluating sigmoid gradient...\n");
  arma::rowvec sigmoid_gradient_vec = arma::ones<arma::rowvec>(5);
  sigmoid_gradient_vec(1) = -0.5;
  sigmoid_gradient_vec(2) = 0;
  sigmoid_gradient_vec(3) = 0.5;
  const arma::mat kSigmoidGradient = \
    neu_net.ComputeSigmoidGradient(sigmoid_gradient_vec);
  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(6);
  printf("Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n");
  kSigmoidGradient.raw_print(std::cout);
  printf("\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Trains neural network.
  printf("Training Neural Network...\n");
  const int kReturnCode = neu_net.RandInitializeWeights();
  arma::vec theta1_init_stack = arma::vectorise(neu_net.theta().at(0));
  arma::vec theta2_init_stack = arma::vectorise(neu_net.theta().at(1));
  arma::vec theta_init_stack = \
    arma::join_vert(theta1_init_stack,theta2_init_stack);
  std::vector<double> theta_init_stack_vec;
  for(unsigned int unit_idx=0; unit_idx<theta_init_stack.n_rows; unit_idx++)
  {
    theta_init_stack_vec.push_back(as_scalar(theta_init_stack.row(unit_idx)));
  }
  nlopt::opt opt(nlopt::LD_LBFGS,theta_init_stack.n_rows);
  WrapperStruct wrap_struct;
  wrap_struct.neu_net = &neu_net;
  wrap_struct.data_multi = &digit_data;
  opt.set_min_objective(ComputeCostWrapper,&wrap_struct);
  opt.set_stopval(0.46);
  double min_cost = 0.0;
  nlopt::result nlopt_result = opt.optimize(theta_init_stack_vec,min_cost);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Performs one-versus-all classification using trained parameters.
  const int kReturnCode2 = neu_net.LabelPrediction(digit_data);
  const arma::vec trainingPredict = neu_net.predictions();
  const arma::vec trainingLabels = digit_data.training_labels();
  int num_train_match = 0;
  for(int example_index=0; example_index<kNumTrainEx; example_index++)
  {
    if (trainingPredict(example_index) == trainingLabels(example_index))
    {
      num_train_match++;
    }
  }
  printf("\n");
  printf("Training Set Accuracy: %.6f\n",(100.0*num_train_match/kNumTrainEx));

  return 0;
}
