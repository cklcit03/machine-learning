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

// LogisticRegression class functions implement modules that are then passed
// to an unconstrained minimization algorithm.
// RegularizedLogisticRegression class functions re-implement modules from
// LogisticRegression class, accounting for a regularization parameter
// "lambda".
// MultiClassRegularizedLogisticRegression class functions re-implement modules
// from RegularizedLogisticRegression class, accounting for the case where we
// have multiple class labels.

#include "logistic_regression.h"

// This function is applied to vectors.
arma::vec LogisticRegression::ComputeSigmoid(const arma::vec sigmoid_arg) {
  const arma::vec kSigmoid = 1/(1+arma::exp(-sigmoid_arg));

  return kSigmoid;
}

// Arguments "theta" and "grad" are updated by nlopt.
// "theta" corresponds to the optimization parameters.
// "grad" corresponds to the gradient.
double LogisticRegression::ComputeCost(const std::vector<double> &theta,
    std::vector<double> &grad,const Data &data) {
  const int kNumFeatures = data.num_features();
  arma::vec nlopt_theta = arma::randu<arma::vec>(kNumFeatures+1,1);

  // Use current value of "theta" from nlopt.
  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
    nlopt_theta(feature_index) = theta[feature_index];
  }
  set_theta(arma::reshape(nlopt_theta,(kNumFeatures+1),1));
  const arma::mat kTrainingFeatures = data.training_features();
  const arma::vec kTrainingLabels = data.training_labels();
  const arma::vec kSigmoidArg = kTrainingFeatures*theta_.col(0);
  const arma::vec kSigmoidVal = ComputeSigmoid(kSigmoidArg);
  const int kNumTrainEx = data.num_train_ex();
  assert(kNumTrainEx > 0);
  const arma::vec kCostFuncVal = (-1)*(kTrainingLabels%arma::log(kSigmoidVal)+\
    (1-kTrainingLabels)%arma::log(1-kSigmoidVal));
  const double kJTheta = sum(kCostFuncVal)/kNumTrainEx;

  // Update "grad" for nlopt.
  const int kReturnCode = this->ComputeGradient(data);
  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
    grad[feature_index] = gradient_(feature_index);
  }

  return kJTheta;
}

// The number of training examples should always be a positive integer.
int LogisticRegression::ComputeGradient(const Data &data) {
  const int kNumTrainEx = data.num_train_ex();
  assert(kNumTrainEx > 0);
  const arma::mat kTrainingFeatures = data.training_features();
  const int kNumFeatures = data.num_features();
  assert(kNumFeatures > 0);
  const arma::vec kTrainingLabels = data.training_labels();

  const arma::vec kSigmoidArg = kTrainingFeatures*theta_.col(0);
  const arma::vec kSigmoidVal = ComputeSigmoid(kSigmoidArg);

  arma::vec gradient_array = arma::zeros<arma::vec>(kNumFeatures+1);

  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
    const arma::vec gradient_term = \
      (kSigmoidVal-kTrainingLabels) % kTrainingFeatures.col(feature_index);
    gradient_array(feature_index) = sum(gradient_term)/kNumTrainEx;
  }
  set_gradient(gradient_array);

  return 0;
}

// The number of training examples should always be a positive integer.
int LogisticRegression::LabelPrediction(const Data &data) {
  const int kNumTrainEx = data.num_train_ex();
  assert(kNumTrainEx > 0);
  const arma::mat kTrainingFeatures = data.training_features();
  const arma::vec kSigmoidArg = kTrainingFeatures*theta_.col(0);
  const arma::vec kSigmoidVal = ComputeSigmoid(kSigmoidArg);
  for(int example_index=0; example_index<kNumTrainEx; example_index++)
  {
    predictions_(example_index) = (kSigmoidVal(example_index) >= 0.5) ? 1 : 0;
  }

  return 0;
}

// Arguments "opt_param" and "grad" are updated by nlopt.
// "opt_param" corresponds to the optimization parameters.
// "grad" corresponds to the gradient.
double RegularizedLogisticRegression::ComputeCost(\
    const std::vector<double> &opt_param,std::vector<double> &grad,
    const Data &data) {
  const int kNumFeatures = data.num_features();
  arma::vec nlopt_theta = arma::randu<arma::vec>(kNumFeatures+1,1);

  // Use current value of "theta" from nlopt.
  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
    nlopt_theta(feature_index) = opt_param[feature_index];
  }
  set_theta(arma::reshape(nlopt_theta,(kNumFeatures+1),1));
  const arma::mat kTrainingFeatures = data.training_features();
  const arma::vec kTrainingLabels = data.training_labels();
  const arma::vec kCurrentTheta = theta().col(0);
  const arma::vec kSigmoidArg = kTrainingFeatures*kCurrentTheta;
  const arma::vec kSigmoidVal = ComputeSigmoid(kSigmoidArg);
  const int kNumTrainEx = data.num_train_ex();
  assert(kNumTrainEx > 0);
  const arma::vec kCostFuncVal = (-1)*(kTrainingLabels%arma::log(kSigmoidVal)+\
    (1-kTrainingLabels)%arma::log(1-kSigmoidVal));
  const double kJTheta = sum(kCostFuncVal)/kNumTrainEx;
  const arma::vec kThetaSquared = kCurrentTheta%kCurrentTheta;
  const double kRegTerm = (lambda_/(2*kNumTrainEx))*sum(kThetaSquared);
  const double kJThetaReg = kJTheta+kRegTerm;

  // Update "grad" for nlopt.
  const int kReturnCode = this->ComputeGradient(data);
  const arma::vec kCurrentGradient = gradient();
  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
    grad[feature_index] = kCurrentGradient(feature_index);
  }

  return kJThetaReg;
}

// The number of training examples should always be a positive integer.
int RegularizedLogisticRegression::ComputeGradient(const Data &data) {
  const int kNumTrainEx = data.num_train_ex();
  assert(kNumTrainEx > 0);
  const arma::mat kTrainingFeatures = data.training_features();
  const int kNumFeatures = data.num_features();
  assert(kNumFeatures > 0);
  const arma::vec kTrainingLabels = data.training_labels();
  const arma::vec kCurrentTheta = theta().col(0);
  const arma::vec kSigmoidArg = kTrainingFeatures*kCurrentTheta;
  const arma::vec kSigmoidVal = ComputeSigmoid(kSigmoidArg);

  arma::vec gradient_array = arma::zeros<arma::vec>(kNumFeatures+1);
  arma::vec gradient_array_reg = arma::zeros<arma::vec>(kNumFeatures+1);

  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
    const arma::vec gradient_term = \
      (kSigmoidVal-kTrainingLabels) % kTrainingFeatures.col(feature_index);
    gradient_array(feature_index) = sum(gradient_term)/kNumTrainEx;
    gradient_array_reg(feature_index) = gradient_array(feature_index)+\
      (lambda_/kNumTrainEx)*(kCurrentTheta(feature_index));
  }
  gradient_array_reg(0) -= (lambda_/kNumTrainEx)*(kCurrentTheta(0));
  set_gradient(gradient_array_reg);

  return 0;
}

// Unpacks WrapperStruct to obtain instances of 
// MultiClassRegularizedLogisticRegression and Data along with class label.
double ComputeCostWrapper(const std::vector<double> &opt_param,
    std::vector<double> &grad,void *void_data) {
  WrapperStruct *wrap_struct = static_cast<WrapperStruct *>(void_data);
  MultiClassRegularizedLogisticRegression *mul_class_reg_log_reg = \
    wrap_struct->mul_class_reg_log_reg;
  DataMulti *data_multi = wrap_struct->data_multi;
  int kClassLabel = wrap_struct->class_label;

  return mul_class_reg_log_reg->ComputeCost(opt_param,grad,*data_multi,\
    kClassLabel);
}

// This function is applied to matrices.
arma::mat MultiClassRegularizedLogisticRegression::ComputeSigmoid\
  (const arma::mat sigmoid_arg) {
  const arma::mat kSigmoid = 1/(1+arma::exp(-sigmoid_arg));

  return kSigmoid;
}

// Arguments "opt_param" and "grad" are updated by nlopt.
// "opt_param" corresponds to the optimization parameters.
// "grad" corresponds to the gradient.
double MultiClassRegularizedLogisticRegression::ComputeCost(\
    const std::vector<double> &opt_param,std::vector<double> &grad,
    const DataMulti &data_multi,const int &class_label) {
  const int kNumFeatures = data_multi.num_features();
  arma::vec nlopt_theta = arma::randu<arma::vec>(kNumFeatures+1,1);

  // Use current value of "theta" from nlopt.
  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
    nlopt_theta(feature_index) = opt_param[feature_index];
  }
  const arma::vec kCurrentTheta = nlopt_theta;
  arma::mat current_theta_mat = theta();
  current_theta_mat.col(class_label%data_multi.num_labels()) = \
    arma::reshape(kCurrentTheta,(kNumFeatures+1),1);
  set_theta(current_theta_mat);
  const arma::mat kTrainingFeatures = data_multi.training_features();
  const int kNumTrainEx = data_multi.num_train_ex();
  const arma::umat kTrainingLabelsBool = \
    (data_multi.training_labels() == class_label*arma::ones(kNumTrainEx,1));
  const arma::vec kTrainingLabels = \
    arma::conv_to<arma::vec>::from(kTrainingLabelsBool);
  const arma::vec kSigmoidArg = kTrainingFeatures*kCurrentTheta;
  const arma::vec kSigmoidVal = ComputeSigmoid(kSigmoidArg);
  assert(kNumTrainEx > 0);
  const arma::vec kCostFuncVal = (-1)*(kTrainingLabels%arma::log(kSigmoidVal)+\
    (1-kTrainingLabels)%arma::log(1-kSigmoidVal));
  const double kJTheta = sum(kCostFuncVal)/kNumTrainEx;
  const arma::vec kThetaSquared = kCurrentTheta%kCurrentTheta;
  const double kRegTerm = (lambda()/(2*kNumTrainEx))*sum(kThetaSquared);
  const double kJThetaReg = kJTheta+kRegTerm;

  // Update "grad" for nlopt.
  const int kReturnCode = this->ComputeGradient(data_multi,class_label);
  const arma::vec kCurrentGradient = gradient();
  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
    grad[feature_index] = kCurrentGradient(feature_index);
  }

  return kJThetaReg;
}

// The number of training examples should always be a positive integer.
int MultiClassRegularizedLogisticRegression::ComputeGradient(\
  const DataMulti &data_multi,const int &class_label) {
  const int kNumTrainEx = data_multi.num_train_ex();
  assert(kNumTrainEx > 0);
  const arma::mat kTrainingFeatures = data_multi.training_features();
  const int kNumFeatures = data_multi.num_features();
  assert(kNumFeatures > 0);
  const arma::umat kTrainingLabelsBool = \
    (data_multi.training_labels() == class_label*arma::ones(kNumTrainEx,1));
  const arma::vec kTrainingLabels = \
    arma::conv_to<arma::vec>::from(kTrainingLabelsBool);
  const arma::vec kCurrentTheta = \
    theta().col(class_label%data_multi.num_labels());
  const arma::vec kSigmoidArg = kTrainingFeatures*kCurrentTheta;
  const arma::vec kSigmoidVal = ComputeSigmoid(kSigmoidArg);

  arma::vec gradient_array = arma::zeros<arma::vec>(kNumFeatures+1);
  arma::vec gradient_array_reg = arma::zeros<arma::vec>(kNumFeatures+1);

  for(int feature_index=0; feature_index<(kNumFeatures+1); feature_index++)
  {
    const arma::vec gradient_term = \
      (kSigmoidVal-kTrainingLabels) % kTrainingFeatures.col(feature_index);
    gradient_array(feature_index) = sum(gradient_term)/kNumTrainEx;
    gradient_array_reg(feature_index) = gradient_array(feature_index)+\
      (lambda()/kNumTrainEx)*(kCurrentTheta(feature_index));
  }
  gradient_array_reg(0) -= (lambda()/kNumTrainEx)*(kCurrentTheta(0));
  set_gradient(gradient_array_reg);

  return 0;
}

int MultiClassRegularizedLogisticRegression::OneVsAll(DataMulti &data_multi) {
  WrapperStruct wrap_struct;
  wrap_struct.mul_class_reg_log_reg = this;
  wrap_struct.data_multi = &data_multi;

  // Loop over all classes.
  for (int class_index=1; class_index<=data_multi.num_labels(); class_index++)
  {
    printf("class_index = %d\n",class_index);
    wrap_struct.class_label = class_index;
	nlopt::opt opt(nlopt::LD_LBFGS,data_multi.num_features()+1);
    opt.set_min_objective(ComputeCostWrapper,&wrap_struct);
    opt.set_ftol_abs(1e-6);
    std::vector<double> nlopt_theta(data_multi.num_features()+1,0.0);
    double min_cost = 0.0;
    nlopt::result nlopt_result = opt.optimize(nlopt_theta,min_cost);
  }

  return 0;
}

// The number of training examples should always be a positive integer.
int MultiClassRegularizedLogisticRegression::LabelPrediction\
  (const DataMulti &data_multi) {
  const int kNumTrainEx = data_multi.num_train_ex();
  assert(kNumTrainEx > 0);
  const arma::mat kTrainingFeatures = data_multi.training_features();
  const arma::mat kSigmoidArg = kTrainingFeatures*theta();
  const arma::mat kSigmoidVal = ComputeSigmoid(kSigmoidArg);
  arma::vec current_predictions = predictions();
  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(1);
  for(int example_index=0; example_index<kNumTrainEx; example_index++)
  {
    int curr_max_index = 0;
    double curr_max_val = kSigmoidVal.row(example_index)(0);
    for(int label_index=1; label_index<data_multi.num_labels(); label_index++)
    {
      if (kSigmoidVal.row(example_index)(label_index) > curr_max_val)
      {
        curr_max_val = kSigmoidVal.row(example_index)(label_index);
        curr_max_index = label_index;
      }
    }
    if (curr_max_index == 0)
    {
      curr_max_index = 10;
    }
    current_predictions(example_index) = curr_max_index;
  }
  set_predictions(current_predictions);

  return 0;
}