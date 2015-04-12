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

// Machine Learning
// Principal Component Analysis (PCA)
// Problem: Use PCA for dimensionality reduction

#include "pca.h"

int main(void) {
  const std::string kExercise7Data1FileName = "../../ex7data1.txt";
  DataUnlabeledNormalized exercise_7_data_1(kExercise7Data1FileName);

  // Run PCA on input data.
  printf("Running PCA on example dataset.\n");
  const int kReturnCode1 = exercise_7_data_1.FeatureNormalize();
  PCA prin_comp_anal;
  prin_comp_anal.Run(exercise_7_data_1);
  printf("Top eigenvector: \n");
  printf("U(:,1) = %f %f\n",prin_comp_anal.left_sing_vec().at(0,0),\
    prin_comp_anal.left_sing_vec().at(1,0));
  printf("(you should expect to see -0.707107 -0.707107)\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Project data onto reduced-dimensional space.
  printf("Dimension reduction on example dataset.\n");
  const int kNumDim = 1;
  const int kReturnCode2 = prin_comp_anal.ProjectData(exercise_7_data_1,\
    kNumDim);
  printf("Projection of the first example: %f\n",\
    prin_comp_anal.mapped_data().at(0,0));
  printf("(this value should be about 1.481274)\n");

  // Maps projected data back onto original space.
  const int kReturnCode3 = prin_comp_anal.RecoverData(exercise_7_data_1,\
    kNumDim);
  printf("Approximation of the first example: %f %f\n",\
    prin_comp_anal.recovered_data().at(0,0),\
    prin_comp_anal.recovered_data().at(0,1));
  printf("(this value should be about  -1.047419 -1.047419)\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  return 0;
}
