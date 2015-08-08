// Copyright (C) 2015  Caleb Lo
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
// Programming Exercise 6: Support Vector Machines
// Problem: Use SVMs to learn decision boundaries for various example datasets

#include "dataset3Params.h"
#include "support_vector_machine.h"

int main(void) {
  printf("Loading Data ...\n");
  const std::string kExampleData1FileName = "../../exampleData1.txt";
  Data example_data_1(kExampleData1FileName);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Trains linear SVM on data.
  printf("Training Linear SVM ...\n");
  const int kSvmType1 = C_SVC;
  const int kKernelType1 = LINEAR;
  SupportVectorMachine svm_model_1(example_data_1,kSvmType1,kKernelType1,0.5,1.0);
  svm_model_1.Train();
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Loads another dataset.
  printf("Loading Data ...\n");
  const std::string kExampleData2FileName = "../../exampleData2.txt";
  Data example_data_2(kExampleData2FileName);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Trains radial basis SVM on data.
  printf("Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n");
  const int kSvmType2 = C_SVC;
  const int kKernelType2 = RBF;
  const double kGamma2 = 1.0/(2.0*0.1*0.1);
  SupportVectorMachine svm_model_2(example_data_2,kSvmType2,kKernelType2,\
    kGamma2,1.0);
  svm_model_2.Train();
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Loads another dataset.
  printf("Loading Data ...\n");
  const std::string kExampleData3FileName = "../../exampleData3.txt";
  const std::string kExampleValData3FileName = "../../exampleValData3.txt";

  // For this exercise, we will use our validation data as test data.
  DataDebug example_data_3(kExampleData3FileName,kExampleValData3FileName,\
    kExampleValData3FileName);

  // Trains radial basis SVM on data.
  // Uses cross-validation to select optimal C and gamma parameters.
  arma::rowvec C_vec = arma::ones<arma::rowvec>(8);
  C_vec(0) = 0.01;
  C_vec(1) = 0.03;
  C_vec(2) = 0.1;
  C_vec(3) = 0.3;
  C_vec(4) = 1.0;
  C_vec(5) = 3.0;
  C_vec(6) = 10.0;
  C_vec(7) = 30.0;
  arma::rowvec sigma_vec = arma::ones<arma::rowvec>(8);
  sigma_vec(0) = 0.01;
  sigma_vec(1) = 0.03;
  sigma_vec(2) = 0.1;
  sigma_vec(3) = 0.3;
  sigma_vec(4) = 1.0;
  sigma_vec(5) = 3.0;
  sigma_vec(6) = 10.0;
  sigma_vec(7) = 30.0;
  double best_C = 0.0;
  double best_sigma = 0.0;
  const int kReturnCode = Dataset3Params(example_data_3,C_vec,sigma_vec,\
    &best_C,&best_sigma);
  assert(best_sigma > 0.0);
  const int kSvmType3 = C_SVC;
  const int kKernelType3 = RBF;
  const double kGamma3 = 1.0/(2.0*best_sigma*best_sigma);
  SupportVectorMachine svm_model_3(example_data_3,kSvmType3,kKernelType3,\
    kGamma2,best_C);
  svm_model_3.Train();
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  return 0;
}
