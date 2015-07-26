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
// Programming Exercise 1: Linear Regression
// Problem: Predict housing prices given sizes/bedrooms of various houses
// Linear regression with multiple variables

#include "gradient_descent.h"
#include "normal_eqn.h"

int main(void) {
  const double kAlpha = 0.1;
  const int kIterations = 400;
  arma::vec theta_vec = arma::randu<arma::vec>(3,1);
  theta_vec.zeros(3,1);
  GradientDescent grad_des(kAlpha,kIterations,theta_vec);
  printf("Loading data ...\n");
  const std::string kDataFileName = "../../housingData.txt";
  DataNormalized housing_data(kDataFileName);
  printf("First 10 examples from the dataset: \n");
  for(int example_idx=0; example_idx<10; example_idx++)
  {
    printf("x = [%.0f %.0f], y = %.0f\n",\
      housing_data.training_features()(example_idx,1),\
      housing_data.training_features()(example_idx,2),\
      housing_data.training_labels()(example_idx));
  }
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Normalizes training features.
  printf("Normalizing Features ...\n");
  const int kReturnCode1 = housing_data.FeatureNormalize();

  // Computes optimal weights using gradient descent.
  printf("Running gradient descent ...\n");
  const int kReturnCode2 = grad_des.RunGradientDescent(housing_data);
  const arma::vec kThetaFinal = grad_des.theta();
  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(6);
  printf("Theta computed from gradient descent: \n");
  kThetaFinal.raw_print(std::cout);
  printf("\n");

  // Predicts price for a 1650 square-foot house with 3 bedrooms.
  arma::rowvec house_vec_1 = arma::ones<arma::rowvec>(2);
  house_vec_1(0) = 1650;
  house_vec_1(1) = 3;
  const arma::rowvec kHouseVec1Normalized = \
    (house_vec_1-housing_data.mu_vec().t())/housing_data.sigma_vec().t();
  const arma::rowvec kHouseVec1NormalizedAug = \
    arma::join_horiz(arma::ones<arma::vec>(1),kHouseVec1Normalized);
  const double kPredPrice1 = \
    arma::as_scalar(kHouseVec1NormalizedAug*kThetaFinal);
  printf("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n",kPredPrice1);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Computes optimal weights using normal equations.
  printf("Solving with normal equations...\n");
  const arma::vec kThetaNormal = NormalEqn(housing_data);
  printf("Theta computed from the normal equations: \n");
  kThetaNormal.raw_print(std::cout);
  printf("\n");

  // Uses normal equations to predict price for a 1650 square-foot house with 3
  // bedrooms.
  arma::rowvec house_vec_2 = arma::ones<arma::rowvec>(3);
  house_vec_2(1) = 1650;
  house_vec_2(2) = 3;
  const double kPredPrice2 = arma::as_scalar(house_vec_2*kThetaNormal);
  printf("Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n",kPredPrice2);

  return 0;
}
