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

// SupportVectorMachine class functions provide an interface to LibSVM to train
// an SVM and perform class prediction using that SVM.

#include "support_vector_machine.h"

// Calls LibSVM function svm_train for simplicity.
int SupportVectorMachine::Train() {

  // Calls svm_train to train an SVM model.
  const char *svm_check_param = svm_check_parameter(&svm_problem_,\
    &svm_parameter_);
  struct svm_model *curr_svm_model = svm_train(&svm_problem_,&svm_parameter_);
  memcpy(&svm_model_,curr_svm_model,sizeof(struct svm_model));

  return 0;
}

// Calls LibSVM function svm_predict for simplicity.
int SupportVectorMachine::Predict(const DataDebug &data_debug) {
  assert(data_debug.num_test_ex() >= 1);
  assert(data_debug.num_features() >= 1);

  // Calls svm_predict to perform prediction for the trained SVM model.
  // svm_predict works on one test case at a time.
  int num_mismatch = 0;
  for(int test_index=0; test_index<data_debug.num_test_ex(); test_index++)
  {
    struct svm_node *curr_test_ex;
    curr_test_ex = new svm_node[data_debug.num_features()+1];
    int sparse_index = 0;
    for(int feat_index=0; feat_index<data_debug.num_features(); feat_index++)
    {
      if (data_debug.testing_features().at(test_index,feat_index) != 0) {
        curr_test_ex[sparse_index].index = feat_index;
        curr_test_ex[sparse_index].value = \
          data_debug.testing_features().at(test_index,feat_index);
        sparse_index++;
      }
    }

    // The end of each test vector is indicated by setting index to -1.
    curr_test_ex[sparse_index].index = -1;
    curr_test_ex[sparse_index].value = 0;
    double pred_result = svm_predict(&svm_model_,curr_test_ex);
    if (pred_result != data_debug.testing_labels().at(test_index)) {
      num_mismatch++;
    }
    delete [] curr_test_ex;
  }

  return num_mismatch;
}
