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
// Programming Exercise 7: K-Means Clustering
// Problem: Apply K-Means Clustering to image compression

#include "kmeans_clustering.h"

int main(void) {
  printf("Finding closest centroids.\n");
  const std::string kExercise7Data2FileName = "../../ex7data2.txt";
  DataUnlabeled exercise_7_data_2(kExercise7Data2FileName);

  // Selects an initial set of centroids.
  int num_centroids = 3;
  arma::mat initial_centroids = arma::zeros<arma::mat>(3,2);
  initial_centroids(0,0) = 3.0;
  initial_centroids(0,1) = 3.0;
  initial_centroids(1,0) = 6.0;
  initial_centroids(1,1) = 2.0;
  initial_centroids(2,0) = 8.0;
  initial_centroids(2,1) = 5.0;

  // Finds closest centroids for example data using initial centroids.
  KMeansClustering k_means_cluster(exercise_7_data_2,num_centroids,\
    initial_centroids);
  const int kReturnCode = k_means_cluster.FindClosestCentroids(exercise_7_data_2);
  printf("Closest centroids for the first 3 examples:\n");
  printf("%d %d %d\n",(int)k_means_cluster.centroid_assignments().at(0),\
    (int)k_means_cluster.centroid_assignments().at(1),\
    (int)k_means_cluster.centroid_assignments().at(2));
  printf("(the closest centroids should be 1, 3, 2 respectively)\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Updates centroids for example data.
  printf("Computing centroids means.\n");
  const int kReturnCode2 = k_means_cluster.ComputeCentroids(exercise_7_data_2);
  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(6);
  printf("Centroids computed after initial finding of closest centroids:\n");
  k_means_cluster.centroids().row(0).raw_print(std::cout);
  k_means_cluster.centroids().row(1).raw_print(std::cout);
  k_means_cluster.centroids().row(2).raw_print(std::cout);
  printf("   [ 2.428301 3.157924 ]\n");
  printf("   [ 5.813503 2.633656 ]\n");
  printf("   [ 7.119387 3.616684 ]\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Runs K-Means Clustering on an example dataset.
  printf("Running K-Means clustering on example dataset.\n");
  const int kMaxIter = 10;

  // Resets initial centroids.
  k_means_cluster.set_centroids(initial_centroids);
  const int kReturnCode3 = k_means_cluster.Run(exercise_7_data_2,kMaxIter);
  printf("K-Means Done.\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  return 0;
}
