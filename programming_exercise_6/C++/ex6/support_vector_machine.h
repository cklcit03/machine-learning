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

// SupportVectorMachine class 1) provides an interface to LibSVM and 2) stores
// relevant parameters.

#ifndef SUPPORT_VECTOR_MACHINE_H_
#define SUPPORT_VECTOR_MACHINE_H_

#include <assert.h>
#include <string>

#include "svm.h"

#include "data.h"

// Provides an interface to LibSVM and stores relevant parameters.
// Sample usage:
// SupportVectorMachine supp_vec_mach(training_data,svm_type,kernel_type);
// const int kReturnCode = supp_vec_mach.Train();
class SupportVectorMachine
{
 public:
  // Sets default values for SVM problem and its associated parameters.
  SupportVectorMachine() {
    svm_problem_.l = 1;
    svm_problem_.y[0] = 0.0;
    svm_problem_.x[0][0].index = -1;
    svm_problem_.x[0][0].value = 0;
    svm_parameter_.svm_type = C_SVC;
    svm_parameter_.kernel_type = LINEAR;
    svm_parameter_.gamma = 1.0;
    svm_parameter_.C = 1.0;
    svm_parameter_.degree = 3;
    svm_parameter_.cache_size = 100;
    svm_parameter_.eps = 0.001;
    svm_parameter_.shrinking = 1;
    svm_parameter_.probability = 0;
    svm_parameter_.nr_weight = 2;
    svm_parameter_.weight = new double[svm_parameter_.nr_weight];
    svm_parameter_.weight[0] = 1.0;
    svm_parameter_.weight[1] = 1.0;
    svm_parameter_.weight_label = new int[svm_parameter_.nr_weight];
    svm_parameter_.weight_label[0] = 0;
    svm_parameter_.weight_label[1] = 1;
  }

  // Sets values for SVM problem and its associated parameters.
  // "data" corresponds to the training data for the SVM problem.
  // "svm_type_arg" corresponds to the type of SVM
  // "kernel_type_arg" corresponds to the type of kernel function
  // "gamma_arg" corresponds to the gamma parameter for a poly/rbf/sigmoid SVM.
  // "C_arg" corresponds to the C parameter for C_SVC, EPSILON_SVR, and NU_SVR.
  SupportVectorMachine(const Data &data,int svm_type_arg,int kernel_type_arg,\
    double gamma_arg,double C_arg) {

    // Sets the number of training data.
    svm_problem_.l = data.num_train_ex();

    // Sets the target values of the training data and sparse representations
    // of the training vectors.
    svm_problem_.y = new double[svm_problem_.l];
    svm_problem_.x = new svm_node*[svm_problem_.l];
    for(int ex_index=0; ex_index<svm_problem_.l; ex_index++)
    {
      svm_problem_.x[ex_index] = new svm_node[data.num_features()+1];
      svm_problem_.y[ex_index] = data.training_labels()[ex_index];
      int sparse_index = 0;
      for(int feat_index=0; feat_index<data.num_features(); feat_index++)
      {
        if (data.training_features().at(ex_index,feat_index) != 0) {
          svm_problem_.x[ex_index][sparse_index].index = feat_index;
          svm_problem_.x[ex_index][sparse_index].value = \
            data.training_features().at(ex_index,feat_index);
          sparse_index++;
        }
      }

      // The end of each training vector is indicated by setting index to -1.
      svm_problem_.x[ex_index][sparse_index].index = -1;
      svm_problem_.x[ex_index][sparse_index].value = 0;
    }

    // Initializes the parameters for the SVM problem.
    svm_parameter_.svm_type = svm_type_arg;
    svm_parameter_.kernel_type = kernel_type_arg;
    svm_parameter_.gamma = gamma_arg;
    svm_parameter_.C = C_arg;

    // Sets default parameters to ensure validity of svm_parameter_.
    svm_parameter_.degree = 3;
    svm_parameter_.cache_size = 100;
    svm_parameter_.eps = 0.001;
    svm_parameter_.shrinking = 1;
    svm_parameter_.probability = 0;
    svm_parameter_.nr_weight = 2;
    svm_parameter_.weight = new double[svm_parameter_.nr_weight];
    svm_parameter_.weight[0] = 1.0;
    svm_parameter_.weight[1] = 1.0;
    svm_parameter_.weight_label = new int[svm_parameter_.nr_weight];
    svm_parameter_.weight_label[0] = 0;
    svm_parameter_.weight_label[1] = 1;

    // Initializes the SVM model to NULL since it will be trained later.
    memset(&svm_model_,0,sizeof(struct svm_model));
  }

  ~SupportVectorMachine() {
    delete [] svm_parameter_.weight_label;
    delete [] svm_parameter_.weight;
    for(int ex_index=0; ex_index<svm_problem_.l; ex_index++)
    {
      delete [] svm_problem_.x[ex_index];
    }
    delete [] svm_problem_.x;
    delete [] svm_problem_.y;
  }

  // Calls svm_train (from LibSVM) to train an SVM model.
  int Train(void);

  // Calls svm_predict (from LibSVM) to perform prediction using the trained
  // SVM model.
  int Predict(const DataDebug &data_debug);

  inline struct svm_problem svm_problem() const {
    return svm_problem_;
  }

  inline struct svm_parameter svm_parameter() const {
    return svm_parameter_;
  }

  inline struct svm_model svm_model() const {
    return svm_model_;
  }

  inline int set_svm_problem(struct svm_problem svm_problem_arg) {
    svm_problem_ = svm_problem_arg;

    return 0;
  }

  inline int set_svm_parameter(struct svm_parameter svm_parameter_arg) {
    svm_parameter_ = svm_parameter_arg;

    return 0;
  }

  inline int set_svm_model(struct svm_model svm_model_arg) {
    svm_model_ = svm_model_arg;

    return 0;
  }

 private:
  // SVM problem.
  struct svm_problem svm_problem_;

  // Parameters of an SVM model.
  struct svm_parameter svm_parameter_;

  // SVM model.
  struct svm_model svm_model_;

  DISALLOW_COPY_AND_ASSIGN(SupportVectorMachine);
};

#endif	// SUPPORT_VECTOR_MACHINE_H_
