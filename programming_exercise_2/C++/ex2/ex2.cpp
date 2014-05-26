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
// Problem: Predict chances of university admission for an applicant given 
// data for admissions decisions and test scores of various applicants

#include "logistic_regression.h"

int main(void) {
  const int kNumIterations = 400;
  arma::vec theta_vec = arma::randu<arma::vec>(3,1);
  theta_vec.zeros(3,1);
  arma::vec gradient_vec = arma::randu<arma::vec>(3,1);
  gradient_vec.zeros(3,1);
  arma::vec predictions_vec = arma::randu<arma::vec>(100,1);
  LogisticRegression log_res(kNumIterations,theta_vec,gradient_vec,\
    predictions_vec);
  const std::string kDataFileName = "../../applicantData.txt";
  Data applicant_data(kDataFileName);

  // Compute initial cost and gradient.
  const std::vector<double> kTheta(3,0.0);
  std::vector<double> grad(3,0.0);
  const double kInitCost = log_res.ComputeCost(kTheta,grad,applicant_data);
  const int kReturnCode = log_res.ComputeGradient(applicant_data);
  const arma::vec kInitGradient = log_res.gradient();
  printf("Cost at initial theta (zeros): %.6f\n",kInitCost);
  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(6);
  printf("Gradient at initial theta (zeros): \n");
  kInitGradient.raw_print(std::cout);
  printf("\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Use L-BFGS algorithm to solve for optimum weights and cost.
  nlopt::opt opt(nlopt::LD_LBFGS,3);
  WrapperStruct wrap_struct;
  wrap_struct.log_res = &log_res;
  wrap_struct.data = &applicant_data;
  opt.set_min_objective(ComputeCostWrapper,&wrap_struct);
  opt.set_xtol_rel(1e-4);
  std::vector<double> nlopt_theta(3,0.0);
  double min_cost = 0.0;
  nlopt::result nlopt_result = opt.optimize(nlopt_theta,min_cost);
  const int kNumFeatures = applicant_data.num_features();
  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
    theta_vec(feature_index) = nlopt_theta[feature_index];
  }
  printf("Cost at theta found by nlopt: %.6f\n",min_cost);
  printf("theta:\n");
  theta_vec.raw_print(std::cout);
  printf("\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Predict admission probability for a student with score 45 on exam 1 and 
  // score 85 on exam 2.
  arma::vec kStudentScores = arma::randu<arma::vec>(3,1);
  kStudentScores(0) = 1;
  kStudentScores(1) = 45;
  kStudentScores(2) = 85;
  const double kAdmissionProb = \
    as_scalar(log_res.ComputeSigmoid(kStudentScores.t()*theta_vec));
  printf("For a student with scores 45 and 85, we predict an admission probability of %.6f\n",\
    kAdmissionProb);
  printf("\n");

  // Compute accuracy on training set.
  const int kReturnCode2 = log_res.LabelPrediction(applicant_data);
  const int kNumTrainEx = applicant_data.num_train_ex();
  const arma::vec trainingPredict = log_res.predictions();
  const arma::vec trainingLabels = applicant_data.training_labels();
  int num_train_match = 0;
  for(int example_index=0; example_index<kNumTrainEx; example_index++)
  {
    if (trainingPredict(example_index) == trainingLabels(example_index))
	{
	  num_train_match++;
	}
  }
  printf("Train Accuracy: %.6f\n",(100.0*num_train_match/kNumTrainEx));
  printf("\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  return 0;
}
