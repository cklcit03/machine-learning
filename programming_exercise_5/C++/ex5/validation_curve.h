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

// Declares function that generates values for a cross-validation curve.

#ifndef VALIDATION_CURVE_H_
#define VALIDATION_CURVE_H_

#include "armadillo"

#include "data.h"
#include "linear_regression.h"

int ValidationCurve(DataDebug &data_debug,LinearRegression &lin_reg,\
  double *error_train,double *error_val,arma::rowvec lambda_vec);

#endif  // VALIDATION_CURVE_H
