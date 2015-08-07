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
// DataDebug class inherits from Data class; it also stores cross-validation 
// data and testing data.

#ifndef MACHINE_LEARNING_PROGRAMMING_EXERCISE_5_EX5_DATA_H_
#define MACHINE_LEARNING_PROGRAMMING_EXERCISE_5_EX5_DATA_H_

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
  // For linear regression, always includes a dummy feature (that is set to 
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

// Stores training data, cross-validation data and testing data.  Sample usage:
// const std::string kTrainDataFileName = "trainData.txt";
// const std::string kValDataFileName = "valData.txt";
// const std::string kTestDataFileName = "testData.txt";
// DataDebug data_debug(kTrainDataFileName,kValDataFileName,kTestDataFileName);
class DataDebug: public Data
{
 public:
  // Sets default values for training data.
  DataDebug() : Data() {}

  // Reads three CSV files: "train_file_name_arg", "val_file_name_arg" and
  // "test_file_name_arg".
  // Sets values for training data based on contents of "train_file_name_arg".
  // Sets values for cross-validation data based on contents of
  // "val_file_name_arg".
  // Sets values for testing data based on contents of "test_file_name_arg".
  explicit DataDebug(std::string train_file_name_arg,\
    std::string val_file_name_arg,std::string test_file_name_arg) : \
    Data(train_file_name_arg) {
    arma::mat validation_data;
    validation_data.load(val_file_name_arg,arma::csv_ascii);
    num_val_ex_ = validation_data.n_rows;
    validation_features_ = \
      arma::join_horiz(arma::ones<arma::vec>(num_val_ex_),\
        validation_data.cols(0,num_features()-1));
    validation_labels_ = validation_data.col(num_features());
    arma::mat testing_data;
    testing_data.load(test_file_name_arg,arma::csv_ascii);
    num_test_ex_ = testing_data.n_rows;
    testing_features_ = \
      arma::join_horiz(arma::ones<arma::vec>(num_test_ex_),\
        testing_data.cols(0,num_features()-1));
    testing_labels_ = testing_data.col(num_features());
  }

  ~DataDebug() {}

  // Performs feature mapping for polynomial regression.
  // Number of polynomial features is "degree_arg".
  // Applies to current set of training features, testing features and
  // cross-validation features.
  // Each column of mapped data is obtained by raising a column vector of
  // features to a given power.
  int PolyFeatures(int degree_arg);

  // Computes mean and standard deviation of each training feature.
  // Computes normalized training data using following formula: 
  // (normalized data) = ((raw data) - (feature mean)) / (feature standard deviation)
  // Normalizes testing data and cross-validation data using the
  // above-mentioned mean and standard deviation.
  int FeatureNormalize();

  inline virtual arma::mat features() const {
    return features_;
  }
  
  inline virtual arma::mat testing_features() const {
    return testing_features_;
  }

  inline virtual arma::mat validation_features() const {
    return validation_features_;
  }

  inline virtual arma::vec labels() const {
    return labels_;
  }

  inline virtual arma::vec testing_labels() const {
    return testing_labels_;
  }

  inline virtual arma::vec validation_labels() const {
    return validation_labels_;
  }

  inline virtual int num_test_ex() const {
    return num_test_ex_;
  }

  inline virtual int num_val_ex() const {
    return num_val_ex_;
  }

  inline virtual arma::mat features_poly() const {
    return features_poly_;
  }
  
  inline virtual arma::mat testing_features_poly() const {
    return testing_features_poly_;
  }

  inline virtual arma::mat validation_features_poly() const {
    return validation_features_poly_;
  }

  inline arma::mat features_normalized() const {
    return features_normalized_;
  }

  inline arma::mat testing_features_normalized() const {
    return testing_features_normalized_;
  }

  inline arma::mat validation_features_normalized() const {
    return validation_features_normalized_;
  }

  inline arma::mat mu_vec() const {
    return mu_vec_;
  }

  inline arma::mat sigma_vec() const {
    return sigma_vec_;
  }

  inline virtual int set_features(arma::mat features_arg) {
    features_ = features_arg;

    return 0;
  }

  inline virtual int set_testing_features(arma::mat testing_features_arg) {
    testing_features_ = testing_features_arg;

    return 0;
  }

  inline virtual int set_validation_features(arma::mat validation_features_arg) {
    validation_features_ = validation_features_arg;

    return 0;
  }

  inline virtual int set_labels(arma::vec labels_arg) {
    labels_ = labels_arg;

    return 0;
  }

  inline virtual int set_testing_labels(arma::vec testing_labels_arg) {
    testing_labels_ = testing_labels_arg;

    return 0;
  }

  inline virtual int set_validation_labels(arma::vec validation_labels_arg) {
    validation_labels_ = validation_labels_arg;

    return 0;
  }

  inline virtual int set_num_test_ex(int num_test_ex_arg) {
    num_test_ex_ = num_test_ex_arg;

    return 0;
  }

  inline virtual int set_num_val_ex(int num_val_ex_arg) {
    num_val_ex_ = num_val_ex_arg;

    return 0;
  }

  inline virtual int set_features_poly(arma::mat features_poly_arg) {
    features_poly_ = features_poly_arg;

    return 0;
  }

  inline virtual int set_testing_features_poly(arma::mat testing_features_poly_arg) {
    testing_features_poly_ = testing_features_poly_arg;

    return 0;
  }

  inline virtual int set_validation_features_poly(arma::mat validation_features_poly_arg) {
    validation_features_poly_ = validation_features_poly_arg;

    return 0;
  }

  inline int set_features_normalized(arma::mat features_normalized_arg) {
    features_normalized_ = features_normalized_arg;

    return 0;
  }

  inline int set_testing_features_normalized(arma::mat testing_features_normalized_arg) {
    testing_features_normalized_ = testing_features_normalized_arg;

    return 0;
  }

  inline int set_validation_features_normalized(arma::mat validation_features_normalized_arg) {
    validation_features_normalized_ = validation_features_normalized_arg;

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

  // Features that are used for computing cost and gradient.
  arma::mat features_;

  // Test and cross-validation features.
  arma::mat testing_features_;
  arma::mat validation_features_;

  // Labels that are used for computing cost and gradient.
  arma::vec labels_;

  // Test and cross-validation labels.
  arma::vec testing_labels_;
  arma::vec validation_labels_;

  // Number of test and cross-validation examples.
  int num_test_ex_;
  int num_val_ex_;

  // Polynomial training, test and cross-validation features.
  arma::mat features_poly_;
  arma::mat testing_features_poly_;
  arma::mat validation_features_poly_;

  // Normalized training, test and cross-validation features.
  arma::mat features_normalized_;
  arma::mat testing_features_normalized_;
  arma::mat validation_features_normalized_;

  // Mean of each training feature.
  arma::vec mu_vec_;

  // Standard deviation of each training feature.
  arma::vec sigma_vec_;

  DISALLOW_COPY_AND_ASSIGN(DataDebug);
};

#endif  // MACHINE_LEARNING_PROGRAMMING_EXERCISE_5_EX5_DATA_H_
