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
// Programming Exercise 6: Spam Classification
// Problem: Use SVMs to determine whether various e-mails are spam 
// (or non-spam)

#include "support_vector_machine.h"

// Defines custom Compare object that will return indices of sorted elements.
bool compare_weights(const std::pair<double,int> &pair1,\
  const std::pair<double,int> &pair2) {
  return (pair1.first < pair2.first);
}

int main(void) {
  printf("Preprocessing sample email (emailSample1.txt)\n");
  const std::string kEmailSample1FileName = "../../emailSample1.txt";
  DataEmail email_sample_1(kEmailSample1FileName);
  assert(email_sample_1.word_indices().size() >= 1);
  printf("Word Indices: \n");
  for(int word_idx=0; word_idx<(int)email_sample_1.word_indices().size(); \
    word_idx++)
  {
    printf("%d\n",email_sample_1.word_indices().at(word_idx));
  }
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();
  printf("Extracting features from sample email (emailSample1.txt)\n");
  printf("Length of feature vector: %d\n",email_sample_1.num_features());
  printf("Number of non-zero entries: %d\n",\
    (int)arma::sum(arma::sum(email_sample_1.features())));
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Trains a linear SVM for spam classification.
  const std::string kSpamTrainFileName = "../../spamTrain.txt";
  DataDebug spam_train(kSpamTrainFileName,kSpamTrainFileName,\
    kSpamTrainFileName);
  assert(spam_train.num_test_ex() >= 1);
  assert(spam_train.num_features() >= 1);
  printf("Training Linear SVM (Spam Classification)\n");
  printf("(this may take 1 to 2 minutes) ...\n");
  const int kSvmType = C_SVC;
  const int kKernelType = LINEAR;
  SupportVectorMachine svm_model(spam_train,kSvmType,kKernelType,0.5,0.1);
  svm_model.Train();

  // Initially, we will use our training data as test data.
  int *spam_train_pred_result = new int[spam_train.num_test_ex()];
  int curr_pred_err = svm_model.Predict(spam_train,spam_train_pred_result);
  printf("Training Accuracy: %.6f\n",\
    100*(1.0-(float)curr_pred_err/(float)spam_train.num_test_ex()));
  delete [] spam_train_pred_result;

  // Tests this linear spam classifier.
  const std::string kSpamTestFileName = "../../spamTest.txt";
  DataDebug spam_test(kSpamTestFileName,kSpamTestFileName,\
    kSpamTestFileName);
  assert(spam_test.num_test_ex() >= 1);
  printf("Evaluating the trained Linear SVM on a test set ...\n");
  int *spam_test_pred_result = new int[spam_test.num_test_ex()];
  curr_pred_err = svm_model.Predict(spam_test,spam_test_pred_result);
  printf("Test Accuracy: %.6f\n",\
    100*(1.0-(float)curr_pred_err/(float)spam_test.num_test_ex()));
  delete [] spam_test_pred_result;
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Determines the top predictors of spam, where predictors are ranked by
  // their weights in the linear classifier.
  std::vector<std::pair<double,int>> svm_weights_indices;
  double *curr_weight = new double[spam_train.num_features()];
  for(int feat_idx=0; feat_idx<spam_train.num_features(); feat_idx++)
  {
    curr_weight[feat_idx] = 0.0;
  }
  assert(svm_model.svm_model().l >= 1);
  for(int sv_idx=0; sv_idx<svm_model.svm_model().l; sv_idx++)
  {
    double curr_coef = svm_model.svm_model().sv_coef[0][sv_idx];
    for(int col_idx=0; col_idx<spam_train.num_features(); col_idx++)
    {
      int curr_idx = svm_model.svm_model().SV[sv_idx][col_idx].index;
      if (curr_idx == -1) {
        break;
      }
      else {
        double curr_val = svm_model.svm_model().SV[sv_idx][col_idx].value;
        curr_weight[curr_idx] += curr_coef*curr_val;
      }
    }
  }
  for(int feat_idx=0; feat_idx<spam_train.num_features(); feat_idx++)
  {
    svm_weights_indices.push_back(std::pair<double,int>(curr_weight[feat_idx],\
      feat_idx));
  }
  delete [] curr_weight;
  std::sort(svm_weights_indices.begin(),svm_weights_indices.end(),\
    compare_weights);
  std::reverse(svm_weights_indices.begin(),svm_weights_indices.end());
  std::vector<std::string> vocab_list;
  const int kReturnCode = email_sample_1.GetVocabList(&vocab_list);
  printf("Top predictors of spam:\n");
  for(int pred_idx=0; pred_idx<15; pred_idx++)
  {
    std::string curr_vocab_word = \
      vocab_list.at(svm_weights_indices.at(pred_idx).second);
    double curr_weight = svm_weights_indices.at(pred_idx).first;
    std::cout << curr_vocab_word << " (" << curr_weight << ")\n";
  }
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Tests this linear spam classifier on another e-mail.
  const std::string kSpamSample1FileName = "../../spamSample1.txt";
  DataEmail spam_sample_1(kSpamSample1FileName);
  assert(spam_sample_1.num_features() >= 1);

  // Writes e-mail features into a file so that our linear classifier can
  // predict whether or not it is spam.
  const std::string kSpamFeatures1FileName = "../../spamSample1_features.txt";
  FILE *spam_file_pointer = fopen(kSpamFeatures1FileName.c_str(),"w");
  for(int feat_idx=0; feat_idx<spam_sample_1.num_features(); feat_idx++)
  {
    int write_file_result = fprintf(spam_file_pointer,"%.6f,",\
      spam_sample_1.features().at(0,feat_idx));
  }
  int write_file_result = fprintf(spam_file_pointer,"1");
  fclose(spam_file_pointer);
  DataDebug spam_features(kSpamFeatures1FileName,kSpamFeatures1FileName,\
    kSpamFeatures1FileName);
  assert(spam_features.num_test_ex() >= 1);
  int *spam_pred_result = new int[spam_features.num_test_ex()];
  int spam_pred_err = svm_model.Predict(spam_features,spam_pred_result);
  printf("Processed spamSample1.txt\n");
  printf("Spam Classification: %d\n",spam_pred_result[0]);
  printf("(1 indicates spam, 0 indicates not spam)\n");
  delete [] spam_pred_result;

  return 0;
}
