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
// DataMapped class inherits from Data class; it maps training features into
// a higher-dimensional space.

#ifndef DATA_H_
#define DATA_H_

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

  // Called by DataMapped constructor to set number of training examples and
  // vector of training labels.
  Data(std::string file_name_arg,int dummy_arg) {
    arma::mat training_data;
    training_data.load(file_name_arg,arma::csv_ascii);
    num_train_ex_ = training_data.n_rows;
    const int kNumFeatures = training_data.n_cols-1;
    training_labels_ = training_data.col(kNumFeatures);
  }

  // Reads CSV file "file_name_arg".
  // Sets values for training data based on contents of this file.
  // Each row of this file is a training example.
  // Each column of this file is a training feature (except for the last 
  // column, which consists of training labels).
  // For logistic regression, always include a dummy feature (that is set to 
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
  arma::mat training_features_;
  arma::vec training_labels_;
  int num_features_;
  int num_train_ex_;

  DISALLOW_COPY_AND_ASSIGN(Data);
};

// Stores training data, including features and labels.  Sample usage:
// const std::string kDataFileName = "trainingData.txt";
// DataMapped training_data_mapped(kDataFileName);
class DataMapped: public Data
{
 public:
  // Sets default values for training data.
  DataMapped() : Data() {}

  // Reads CSV file "file_name_arg".
  // Sets values for training data based on contents of this file.
  // Each row of this file is a training example.
  // Each column of this file is a training feature (except for the last 
  // column, which consists of training labels).
  // For logistic regression, always include a dummy feature (that is set to 
  // unity) for each training example.
  // Map features into all polynomial terms of first two features up to the
  // sixth power.
  explicit DataMapped(std::string file_name_arg) : Data(file_name_arg,0) {
    arma::mat training_data;
    training_data.load(file_name_arg,arma::csv_ascii);
    const arma::vec kTrainingFeature1 = training_data.col(0);
    const arma::vec kTrainingFeature2 = training_data.col(1);
    arma::mat aug_feature_mat = arma::ones<arma::mat>(num_train_ex(),1);
    const int kDegree = 6;
    for(int deg_index1=1; deg_index1<(kDegree+1); deg_index1++)
    {
      for(int deg_index2=0; deg_index2<(deg_index1+1); deg_index2++)
      {
        const arma::vec aug_feature_mat_term1 = \
          arma::pow(kTrainingFeature1,deg_index1-deg_index2);
        const arma::vec aug_feature_mat_term2 = \
          arma::pow(kTrainingFeature2,deg_index2);
        const arma::vec aug_feature_mat_term = \
          aug_feature_mat_term1 % aug_feature_mat_term2;
        aug_feature_mat = \
          arma::join_horiz(aug_feature_mat,aug_feature_mat_term);
      }
    }
    set_num_features(aug_feature_mat.n_cols-1);
    set_training_features(aug_feature_mat);
  }

  ~DataMapped() {}

 private:
  DISALLOW_COPY_AND_ASSIGN(DataMapped);
};

#endif  // DATA_H_
