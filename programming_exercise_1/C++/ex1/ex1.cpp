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
// Problem: Predict profits for a food truck given data for 
// profits/populations of various cities
// Linear regression with one variable

#include "gradient_descent.h"

int main(void) {
  const double kAlpha = 0.01;
  const int kIterations = 1500;
  arma::vec theta_vec = arma::randu<arma::vec>(2,1);
  theta_vec.zeros(2,1);
  GradientDescent grad_des(kAlpha,kIterations,theta_vec);
  const std::string kDataFileName = "../../foodTruckData.txt";
  Data food_truck_data(kDataFileName);

  // Compute squared error given initial weights in theta_vec.
  printf("Running Gradient Descent ...\n");
  const double kInitCost = grad_des.ComputeCost(food_truck_data);
  printf("ans = %.3f\n",kInitCost);

  // Compute optimal weights using gradient descent.
  const int kReturnCode = grad_des.RunGradientDescent(food_truck_data);
  const arma::vec kThetaFinal = grad_des.theta();
  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(6);
  printf("Theta found by gradient descent: ");
  kThetaFinal.t().raw_print(std::cout);
  printf("\n");

  // Predict profit for population size of 35000.
  arma::rowvec population_vec_1 = arma::ones<arma::rowvec>(2);
  population_vec_1(1) = 3.5;
  const double kPredProfit1 = as_scalar(population_vec_1*kThetaFinal);
  const double kPredProfit1Scaled = 10000*kPredProfit1;
  printf("For population = 35,000, we predict a profit of %f\n",\
    kPredProfit1Scaled);

  // Predict profit for population size of 70000.
  arma::rowvec population_vec_2 = arma::ones<arma::rowvec>(2);
  population_vec_2(1) = 7.0;
  const double kPredProfit2 = as_scalar(population_vec_2*kThetaFinal);
  const double kPredProfit2Scaled = 10000*kPredProfit2;
  printf("For population = 70,000, we predict a profit of %f\n",\
    kPredProfit2Scaled);

  return 0;
}
