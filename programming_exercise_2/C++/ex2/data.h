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

// Data class stores training data, including features and labels.

#ifndef MACHINE_LEARNING_PROGRAMMING_EXERCISE_2_EX2_DATA_H_
#define MACHINE_LEARNING_PROGRAMMING_EXERCISE_2_EX2_DATA_H_

#include <string>

#include "armadillo"

#include "macro.h"

// Stores training data, including features and labels.  Sample usage:
// const std::string kDataFileName = "trainingData.txt";
// Data training_data(kDataFileName);
class Data
{
 public:
  // Sets default values for training data.
  Data() {
    num_train_ex_ = 1;
    num_features_ = 1;
    training_features_.ones(1,2);
    training_labels_.ones(1,1);
  }

  // Reads CSV file "file_name_arg".
  // Sets values for training data based on contents of this file.
  // Each row of this file is a training example.
  // Each column of this file is a training feature (except for the last 
  // column, which consists of training labels).
  // For logistic regression, always includes a dummy feature (that is set to 
  // unity) for each training example.
  explicit Data(std::string file_name_arg) {
    arma::mat training_data;
    training_data.load(file_name_arg,arma::csv_ascii);
    num_train_ex_ = training_data.n_rows;
    num_features_ = training_data.n_cols-1;
    training_features_ = \
      arma::join_horiz(arma::ones<arma::vec>(num_train_ex_),\
        training_data.cols(0,num_features_-1));
    training_labels_ = training_data.col(num_features_);
  }

  ~Data() {}

  inline arma::mat training_features() const {
    return training_features_;
  }

  inline arma::vec training_labels() const {
    return training_labels_;
  }

  inline int num_features() const {
    return num_features_;
  }

  inline int num_train_ex() const {
    return num_train_ex_;
  }

  inline int set_training_features(arma::mat training_features_arg) {
    training_features_ = training_features_arg;

    return 0;
  }

  inline int set_training_labels(arma::vec training_labels_arg) {
    training_labels_ = training_labels_arg;

    return 0;
  }

  inline int set_num_features(int num_features_arg) {
    num_features_ = num_features_arg;

    return 0;
  }

  inline int set_num_train_ex(int num_train_ex_arg) {
    num_train_ex_ = num_train_ex_arg;

    return 0;
  }

 private:
  // Matrix of training features.
  arma::mat training_features_;

  // Vector of training labels.
  arma::vec training_labels_;

  // Number of training features.
  int num_features_;

  // Number of training examples.
  int num_train_ex_;

  DISALLOW_COPY_AND_ASSIGN(Data);
};

#endif  // MACHINE_LEARNING_PROGRAMMING_EXERCISE_2_EX2_DATA_H_
