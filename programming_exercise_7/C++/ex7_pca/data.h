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

// Data class stores training data, including features and labels.
// DataUnlabeled class inherits from Data class; it treats each training 
// example as being unlabeled.
// DataUnlabeledNormalized class inherits from DataUnlabeled class; it includes
// a method for normalizing unlabeled training data.

#ifndef MACHINE_LEARNING_PROGRAMMING_EXERCISE_7_EX7_PCA_DATA_H_
#define MACHINE_LEARNING_PROGRAMMING_EXERCISE_7_EX7_PCA_DATA_H_

#include <assert.h>
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
  explicit Data(std::string file_name_arg) {
    arma::mat training_data;
    training_data.load(file_name_arg,arma::csv_ascii);
    num_train_ex_ = training_data.n_rows;
    num_features_ = training_data.n_cols-1;
    training_features_ = training_data.cols(0,num_features_-1);
    training_labels_ = training_data.col(num_features_);
  }

  ~Data() {}

  inline virtual arma::mat training_features() const {
    return training_features_;
  }

  inline virtual arma::vec training_labels() const {
    return training_labels_;
  }

  inline virtual int num_features() const {
    return num_features_;
  }

  inline virtual int num_train_ex() const {
    return num_train_ex_;
  }

  inline virtual int set_training_features(arma::mat training_features_arg) {
    training_features_ = training_features_arg;

    return 0;
  }

  inline virtual int set_training_labels(arma::vec training_labels_arg) {
    training_labels_ = training_labels_arg;

    return 0;
  }
  
  inline virtual int set_num_features(int num_features_arg) {
    num_features_ = num_features_arg;

    return 0;
  }

  inline virtual int set_num_train_ex(int num_train_ex_arg) {
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

// Treats each training example as being unlabeled.  Sample usage:
// const std::string kDataFileName = "trainingData.txt";
// DataUnlabeled training_data(kDataFileName);
class DataUnlabeled: public Data
{
 public:
  // Sets default values for unlabeled training data.
  DataUnlabeled() {
    num_train_ex_ = 1;
    num_features_ = 2;
    training_features_.ones(1,2);
  }

  // Reads CSV file "file_name_arg".
  // Sets values for unlabeled training data based on contents of this file.
  // Each row of this file is a training example.
  // Each column of this file is a training feature.
  // This file contains no training labels.
  explicit DataUnlabeled(std::string file_name_arg) {
    arma::mat training_data;
    training_data.load(file_name_arg,arma::csv_ascii);
    num_train_ex_ = training_data.n_rows;
    num_features_ = training_data.n_cols;
    training_features_ = training_data.cols(0,num_features_-1);
  }

  ~DataUnlabeled() {}

  inline virtual arma::mat training_features() const {
    return training_features_;
  }

  inline virtual int num_features() const {
    return num_features_;
  }

  inline virtual int num_train_ex() const {
    return num_train_ex_;
  }

  inline virtual int set_training_features(arma::mat training_features_arg) {
    training_features_ = training_features_arg;

    return 0;
  }
  
  inline virtual int set_num_features(int num_features_arg) {
    num_features_ = num_features_arg;

    return 0;
  }

  inline virtual int set_num_train_ex(int num_train_ex_arg) {
    num_train_ex_ = num_train_ex_arg;

    return 0;
  }

 private:
  // Matrix of training features.
  arma::mat training_features_;

  // Number of training features.
  int num_features_;

  // Number of training examples.
  int num_train_ex_;

  DISALLOW_COPY_AND_ASSIGN(DataUnlabeled);
};

// Stores normalized unlabeled training data, including features.  
// Sample usage:
// const std::string kDataFileName = "trainingData.txt";
// DataUnlabeledNormalized training_data_unlabeled_normalized(kDataFileName);
class DataUnlabeledNormalized: public DataUnlabeled
{
 public:
  // Sets default values for normalized unlabeled training data.
  DataUnlabeledNormalized() : DataUnlabeled() {
    training_features_normalized_.ones(1,3);
    mu_vec_.ones(1,2);
    sigma_vec_.ones(1,2);
  }

  // Reads CSV file "file_name_arg".
  // Uses constructor for DataUnlabeled based on contents of this file.
  // Defers initialization of member variables to FeatureNormalize()
  explicit DataUnlabeledNormalized(std::string fileNameArg) : \
    DataUnlabeled(fileNameArg) {}

  ~DataUnlabeledNormalized() {}

  // Computes mean and standard deviation of each training feature.
  // Computes normalized training data using following formula: 
  // (normalized data) = ((raw data) - (feature mean)) / (feature standard deviation)
  int FeatureNormalize();

  inline arma::mat training_features_normalized() const {
    return training_features_normalized_;
  }

  inline arma::mat mu_vec() const {
    return mu_vec_;
  }

  inline arma::mat sigma_vec() const {
    return sigma_vec_;
  }

  inline int set_training_features_normalized(arma::mat training_features_normalized_arg) {
    training_features_normalized_ = training_features_normalized_arg;

    return 0;
  }

  inline int set_mu_vec(arma::vec mu_vec_arg) {
    mu_vec_ = mu_vec_arg;

    return 0;
  }

  inline int set_sigma_vec(arma::vec sigma_vec_arg) {
    sigma_vec_ = sigma_vec_arg;

    return 0;
  }

 private:
  // Matrix of normalized training features.
  arma::mat training_features_normalized_;

  // Mean of each training feature
  arma::vec mu_vec_;

  // Standard deviation of each training feature
  arma::vec sigma_vec_;

  DISALLOW_COPY_AND_ASSIGN(DataUnlabeledNormalized);
};

#endif  // MACHINE_LEARNING_PROGRAMMING_EXERCISE_7_EX7_PCA_DATA_H_
