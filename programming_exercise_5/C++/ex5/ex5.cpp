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
// Programming Exercise 5: Regularized Linear Regression and Bias vs. Variance
// Predict amount of water flowing out of a dam given data for change of water 
// level in a reservoir

#include "learning_curve.h"
#include "linear_regression.h"
#include "validation_curve.h"

int main(void) {
  printf("Loading Data ...\n");
  const std::string kTrainDataFileName = "../../waterTrainData.txt";
  const std::string kValDataFileName = "../../waterValData.txt";
  const std::string kTestDataFileName = "../../waterTestData.txt";
  DataDebug water_data(kTrainDataFileName,kValDataFileName,kTestDataFileName);
  const int kTotFeatures = water_data.num_features()+1;
  arma::vec theta_vec = arma::randu<arma::vec>(kTotFeatures,1);
  theta_vec.zeros(kTotFeatures,1);
  arma::vec gradient_vec = arma::randu<arma::vec>(kTotFeatures,1);
  gradient_vec.zeros(kTotFeatures,1);
  double lambda = 1.0;
  LinearRegression lin_reg(theta_vec,gradient_vec,lambda);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Compute cost for regularized linear regression.
  std::vector<double> theta_stack_vec(kTotFeatures,1.0);
  std::vector<double> grad_vec(kTotFeatures,0.0);
  water_data.set_features(water_data.training_features());
  water_data.set_labels(water_data.training_labels());
  const double kInitCost = \
    lin_reg.ComputeCost(theta_stack_vec,grad_vec,water_data);
  printf("Cost at theta = [1 ; 1]: %.6f\n",kInitCost);
  printf("(this value should be about 303.993192)\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Compute gradient for regularized linear regression.
  const double kReturnCode = lin_reg.ComputeGradient(water_data);
  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(6);
  printf("Gradient at theta = [1 ; 1]: \n");
  lin_reg.gradient().t().raw_print(std::cout);
  printf("(this value should be about [-15.303016; 598.250744])\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Train linear regression.
  const int kReturnCode2 = lin_reg.Train(water_data);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Generate values for learning curve.
  const int kNumTrainEx = water_data.num_train_ex();
  double *error_train = (double *)calloc(kNumTrainEx,sizeof(double));
  double *error_val = (double *)calloc(kNumTrainEx,sizeof(double));
  int use_poly = 0;
  const int kReturnCode3 = \
    LearningCurve(water_data,lin_reg,error_train,error_val,use_poly);
  printf("# Training Examples\tTrain Error\tCross Validation Error\n");
  for(unsigned int ex_index=0; ex_index<(unsigned int)kNumTrainEx; ex_index++)
  {
    printf("\t%d\t\t%.6f\t%.6f\n",ex_index+1,error_train[ex_index],\
      error_val[ex_index]);
  }
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Perform feature mapping for polynomial regression.
  water_data.set_features(water_data.training_features());
  const int kNumPolyFeatures = 8;
  const int kReturnCode4 = water_data.PolyFeatures(kNumPolyFeatures);
  const int kReturnCode5 = water_data.FeatureNormalize();
  printf("Normalized Training Example 1: \n");
  water_data.features_normalized().row(0).t().raw_print(std::cout);
  printf("\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Train polynomial regression.
  water_data.set_features(water_data.features_normalized());
  water_data.set_labels(water_data.training_labels());
  lin_reg.set_lambda(0.0);
  const int kReturnCode6 = lin_reg.Train(water_data);

  // Generate values for learning curve for polynomial regression.
  use_poly = 1;
  const int kReturnCode7 = \
    LearningCurve(water_data,lin_reg,error_train,error_val,use_poly);
  printf("Polynomial Regression (lambda = %.6f)\n",lin_reg.lambda());
  printf("\n");
  printf("# Training Examples\tTrain Error\tCross Validation Error\n");
  for(unsigned int ex_index=0; ex_index<(unsigned int)kNumTrainEx; ex_index++)
  {
    printf("\t%d\t\t%.6f\t%.6f\n",ex_index+1,error_train[ex_index],\
      error_val[ex_index]);
  }
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Generate values for cross-validation curve for polynomial regression.
  // Set up vector of regularization parameters.
  arma::rowvec lambda_vec = arma::ones<arma::rowvec>(10);
  lambda_vec(0) = 0.0;
  lambda_vec(1) = 0.001;
  lambda_vec(2) = 0.003;
  lambda_vec(3) = 0.01;
  lambda_vec(4) = 0.03;
  lambda_vec(5) = 0.1;
  lambda_vec(6) = 0.3;
  lambda_vec(7) = 1.0;
  lambda_vec(8) = 3.0;
  lambda_vec(9) = 10.0;
  double *xval_error_train = (double *)calloc(10,sizeof(double));
  double *xval_error_val = (double *)calloc(10,sizeof(double));
  const int kReturnCode8 = \
    ValidationCurve(water_data,lin_reg,xval_error_train,xval_error_val,lambda_vec);
  printf("lambda\t\tTrain Error\tCross Validation Error\n");
  for(unsigned int lambda_index=0; lambda_index<10; lambda_index++)
  {
    printf("%.6f\t%.6f\t%.6f\n",lambda_vec(lambda_index),\
      xval_error_train[lambda_index],xval_error_val[lambda_index]);
  }
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Free memory.
  free(error_train);
  free(error_val);
  free(xval_error_train);
  free(xval_error_val);

  return 0;
}
