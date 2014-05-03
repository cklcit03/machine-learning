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
// DataNormalized class inherits from Data class; it includes a method for 
// normalizing training data.

#ifndef DATA_H_
#define DATA_H_

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
	num_features_ = 2;
	training_features_.ones(1,3);
	training_labels_.ones(1,1);
  }

  // Reads CSV file "file_name_arg".
  // Sets values for training data based on contents of this file.
  // Each row of this file is a training example.
  // Each column of this file is a training feature (except for the last 
  // column, which consists of training labels).
  // For linear regression, always include a dummy feature (that is set to 
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

 private:
  arma::mat training_features_;
  arma::vec training_labels_;
  int num_features_;
  int num_train_ex_;

  DISALLOW_COPY_AND_ASSIGN(Data);
};

// Stores normalized training data, including features and labels.  
// Sample usage:
// const std::string kDataFileName = "trainingData.txt";
// DataNormalized training_data_normalized(kDataFileName);
class DataNormalized: public Data
{
 public:
  // Sets default values for normalized training data.
  DataNormalized() : Data() {
	training_features_normalized_.ones(1,3);
	mu_vec_.ones(1,2);
	sigma_vec_.ones(1,2);
  }

  // Reads CSV file "file_name_arg".
  // Use constructor for Data based on contents of this file.
  // Defer initialization of member variables to FeatureNormalize()
  explicit DataNormalized(std::string fileNameArg) : Data(fileNameArg) {}

  ~DataNormalized() {}

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
  arma::mat training_features_normalized_;

  // Mean of each training feature
  arma::vec mu_vec_;

  // Standard deviation of each training feature
  arma::vec sigma_vec_;

  DISALLOW_COPY_AND_ASSIGN(DataNormalized);
};

#endif  // DATA_H_
