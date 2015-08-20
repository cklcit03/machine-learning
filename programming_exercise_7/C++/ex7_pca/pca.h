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

// PCA class 1) implements key functions for principal component analysis
// and 2) stores relevant parameters.

#ifndef MACHINE_LEARNING_PROGRAMMING_EXERCISE_7_EX7_PCA_PCA_H_
#define MACHINE_LEARNING_PROGRAMMING_EXERCISE_7_EX7_PCA_PCA_H_

#include <assert.h>
#include <string>

#include "data.h"

// Implements key functions for principal component analysis and stores 
// relevant parameters.
// Sample usage:
// PCA prin_comp_anal();
// const int kReturnCode = prin_comp_anal.Run();
class PCA
{
 public:
  // Sets default values for algorithm parameters.
  PCA() {
    left_sing_vec_ = arma::zeros<arma::mat>(1,1);
    sing_val_ = arma::zeros<arma::vec>(1,1);
    mapped_data_ = arma::zeros<arma::mat>(1,1);
    recovered_data_ = arma::zeros<arma::mat>(1,1);
  }

  ~PCA() {}
  
  // Runs PCA algorithm.
  int Run(const DataUnlabeledNormalized &data_unlabeled_normalized);

  // Projects input data onto reduced-dimensional space.
  int ProjectData(const DataUnlabeledNormalized &data_unlabeled_normalized,\
    int num_dim);

  // Projects input data onto original space.
  int RecoverData(const DataUnlabeledNormalized &data_unlabeled_normalized,\
    int num_dim);

  inline arma::mat left_sing_vec() const {
    return left_sing_vec_;
  }

  inline arma::vec sing_val() const {
    return sing_val_;
  }

  inline arma::mat mapped_data() const {
    return mapped_data_;
  }

  inline arma::mat recovered_data() const {
    return recovered_data_;
  }

  inline int set_left_sing_vec(arma::mat left_sing_vec_arg) {
    left_sing_vec_ = left_sing_vec_arg;

    return 0;
  }

  inline int set_sing_val(arma::vec sing_val_arg) {
    sing_val_ = sing_val_arg;

    return 0;
  }

  inline int set_mapped_data(arma::mat mapped_data_arg) {
    mapped_data_ = mapped_data_arg;

    return 0;
  }

  inline int set_recovered_data(arma::mat recovered_data_arg) {
    recovered_data_ = recovered_data_arg;

    return 0;
  }

 private:
  // Left singular vectors of covariance matrix of (normalized) training data.
  arma::mat left_sing_vec_;

  // Singular values of covariance matrix of (normalized) training data.
  arma::vec sing_val_;

  // (Normalized) training data mapped onto lower-dimensional space.
  arma::mat mapped_data_;

  // Recovered data in original space.
  arma::mat recovered_data_;

  DISALLOW_COPY_AND_ASSIGN(PCA);
};

#endif	// MACHINE_LEARNING_PROGRAMMING_EXERCISE_7_EX7_PCA_PCA_H_
