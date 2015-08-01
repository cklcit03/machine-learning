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
// Programming Exercise 2: Logistic Regression
// Problem: Predict chances of acceptance for a microchip given data for
// acceptance decisions and test scores of various microchips

#include "logistic_regression.h"

int main(void) {
  const std::string kDataFileName = "../../microChipData.txt";
  DataMapped micro_chip_data(kDataFileName);
  const int kNumFeatures = micro_chip_data.num_features();
  arma::vec theta_vec = arma::randu<arma::vec>(kNumFeatures+1,1);
  theta_vec.zeros(kNumFeatures+1,1);
  arma::vec gradient_vec = arma::randu<arma::vec>(kNumFeatures+1,1);
  gradient_vec.zeros(kNumFeatures+1,1);
  const int kNumTrainEx = micro_chip_data.num_train_ex();
  arma::vec predictions_vec = arma::randu<arma::vec>(kNumTrainEx,1);
  const double kLambda = 1.0;
  const int kNumIterations = 400;
  RegularizedLogisticRegression reg_log_reg(kNumIterations,theta_vec,\
    gradient_vec,predictions_vec,kLambda);

  // Computes initial cost.
  const std::vector<double> kTheta(kNumFeatures+1,0.0);
  std::vector<double> grad(kNumFeatures+1,0.0);
  const double kInitCost = reg_log_reg.ComputeCost(kTheta,grad,micro_chip_data);
  printf("Cost at initial theta (zeros): %.6f\n",kInitCost);
  printf("\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Uses MMA algorithm to solve for optimum weights and cost.
  nlopt::opt opt(nlopt::LD_MMA,kNumFeatures+1);
  WrapperStruct wrap_struct;
  wrap_struct.reg_log_reg = &reg_log_reg;
  wrap_struct.data_mapped = &micro_chip_data;
  opt.set_min_objective(ComputeCostWrapper,&wrap_struct);
  opt.set_xtol_rel(1e-4);
  std::vector<double> nlopt_theta(kNumFeatures+1,0.0);
  double min_cost = 0.0;
  nlopt::result nlopt_result = opt.optimize(nlopt_theta,min_cost);
  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
    theta_vec(feature_index) = nlopt_theta[feature_index];
  }

  // Computes accuracy on training set.
  const int kReturnCode2 = reg_log_reg.LabelPrediction(micro_chip_data);
  const arma::vec trainingPredict = reg_log_reg.predictions();
  const arma::vec trainingLabels = micro_chip_data.training_labels();
  int num_train_match = 0;
  for(int example_index=0; example_index<kNumTrainEx; example_index++)
  {
    if (trainingPredict(example_index) == trainingLabels(example_index))
    {
      num_train_match++;
    }
  }
  printf("Train Accuracy: %.6f\n",(100.0*num_train_match/kNumTrainEx));

  return 0;
}
