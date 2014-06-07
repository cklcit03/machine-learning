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
// Programming Exercise 3: (Multi-class) Logistic Regression and 
// Neural Networks
// Problem: Predict label for a handwritten digit given data for pixel values 
// of various handwritten digits
// Use parameters trained by a neural network for prediction

#include "neural_network.h"

int main(void) {
  printf("Loading Data ...\n");
  const std::string kDataFileName = "../../digitData.txt";
  const int kNumLabels = 10;
  DataMulti digit_data(kDataFileName,kNumLabels);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Load two files that contain parameters trained by a neural network.
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

  // Perform one-versus-all classification using trained parameters.
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
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  return 0;
}
