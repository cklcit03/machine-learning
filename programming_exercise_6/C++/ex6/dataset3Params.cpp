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

// Defines function that selects optimal learning parameters for a radial
// basis SVM.

#include "dataset3Params.h"

// Uses LibSVM for training.
int Dataset3Params(DataDebug &data_debug,arma::rowvec C_vec,\
  arma::rowvec sigma_vec,double *C_arg,double *sigma_arg) {
  int best_pred_err = 1000000;
  for(int C_index=0; C_index<8; C_index++)
  {
    for(int sigma_index=0; sigma_index<8; sigma_index++)
	{
      const int kSvmType3 = C_SVC;
      const int kKernelType3 = RBF;
      const double kCurrSigma = sigma_vec(sigma_index);
      const double kGamma3 = 1.0/(2.0*kCurrSigma*kCurrSigma);
	  const double kCurrC = C_vec(C_index);
      SupportVectorMachine svm_model_3(data_debug,kSvmType3,kKernelType3,\
        kGamma3,kCurrC);
      svm_model_3.Train();
	  int curr_pred_err = svm_model_3.Predict(data_debug);
      if (curr_pred_err < best_pred_err) {
        best_pred_err = curr_pred_err;
        *C_arg = kCurrC;
        *sigma_arg = kCurrSigma;
	  }
	}
  }

  return 0;
}
