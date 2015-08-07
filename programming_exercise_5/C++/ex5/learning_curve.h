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

// Declares function that generates values for a learning curve.

#ifndef MACHINE_LEARNING_PROGRAMMING_EXERCISE_5_EX5_LEARNING_CURVE_H_
#define MACHINE_LEARNING_PROGRAMMING_EXERCISE_5_EX5_LEARNING_CURVE_H_

#include "armadillo"

#include "data.h"
#include "linear_regression.h"

int LearningCurve(DataDebug &data_debug,LinearRegression &lin_reg,\
  double *error_train,double *error_val,int use_poly);

#endif  // LEARNING_CURVE_H
