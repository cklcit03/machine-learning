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

// Declares function that selects optimal learning parameters for a radial
// basis SVM.

#ifndef MACHINE_LEARNING_PROGRAMMING_EXERCISE_6_EX6_DATASET3_PARAMS_H_
#define MACHINE_LEARNING_PROGRAMMING_EXERCISE_6_EX6_DATASET3_PARAMS_H_

#include "armadillo"

#include "data.h"
#include "support_vector_machine.h"

int Dataset3Params(DataDebug &data_debug,arma::rowvec C_vec,\
  arma::rowvec sigma_vec,double *C_arg,double *sigma_arg);

#endif  // MACHINE_LEARNING_PROGRAMMING_EXERCISE_6_EX6_DATASET3_PARAMS_H_
