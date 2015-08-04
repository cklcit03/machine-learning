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

#include "logistic_regression.h"

int main(void) {
  printf("Loading Data ...\n");
  const std::string kDataFileName = "../../digitData.txt";
  const int kNumLabels = 10;
  DataMulti digit_data(kDataFileName,kNumLabels);
  const int kNumFeatures = digit_data.num_features();
  arma::mat theta_vec = arma::randu<arma::mat>(kNumFeatures+1,kNumLabels);
  theta_vec.zeros(kNumFeatures+1,kNumLabels);
  arma::vec gradient_vec = arma::randu<arma::vec>(kNumFeatures+1,1);
  gradient_vec.zeros(kNumFeatures+1,1);
  const int kNumTrainEx = digit_data.num_train_ex();
  arma::vec predictions_vec = arma::randu<arma::vec>(kNumTrainEx,1);
  const double kLambda = 0.1;
  const int kNumIterations = 400;
  MultiClassRegularizedLogisticRegression mul_class_reg_log_reg(\
    kNumIterations,theta_vec,gradient_vec,predictions_vec,kLambda,kNumLabels);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Trains one logistic regression classifier for each digit.
  const int kReturnCode = mul_class_reg_log_reg.OneVsAll(digit_data);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Performs one-versus-all classification using logistic regression.
  const int kReturnCode2 = mul_class_reg_log_reg.LabelPrediction(digit_data);
  const arma::vec trainingPredict = mul_class_reg_log_reg.predictions();
  const arma::vec trainingLabels = digit_data.training_labels();
  int num_train_match = 0;
  for(int example_index=0; example_index<kNumTrainEx; example_index++)
  {
    if (trainingPredict(example_index) == trainingLabels(example_index))
    {
      num_train_match++;
    }
  }
  printf("Training Set Accuracy: %.6f\n",(100.0*num_train_match/kNumTrainEx));

  return 0;
}
