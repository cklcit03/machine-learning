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

// KMeansClustering class 1) implements key functions for k-means clustering
// and 2) stores relevant parameters.

#ifndef MACHINE_LEARNING_PROGRAMMING_EXERCISE_7_EX7_KMEANS_CLUSTERING_H_
#define MACHINE_LEARNING_PROGRAMMING_EXERCISE_7_EX7_KMEANS_CLUSTERING_H_

#include <assert.h>
#include <string>

#include "data.h"

// Implements key functions for k-means clustering and stores relevant 
// parameters.
// Sample usage:
// KMeansClustering k_mu_clust(training_data,num_centroids_arg,centroids_arg);
// const int kReturnCode = k_mu_clust.FindClosestCentroids();
class KMeansClustering
{
 public:
  // Sets default values for algorithm parameters.
  KMeansClustering() {
    num_centroids_ = 1;
    centroids_ = arma::zeros<arma::mat>(1,1);
    centroid_assignments_ = arma::zeros<arma::vec>(1,1);
  }

  // Sets values for algorithm parameters.
  // "data_unlabeled" corresponds to the unlabeled training data.
  // "num_centroids_arg" corresponds to the number of centroids.
  // "centroids_arg" corresponds to the centroids.
  KMeansClustering(const DataUnlabeled &data_unlabeled,int num_centroids_arg,\
    arma::mat centroids_arg) {

    // Sets the number of centroids.
    num_centroids_ = num_centroids_arg;

    // Sets the centroids.
    centroids_ = centroids_arg;

    // Sets the centroid assignments.
    centroid_assignments_ = \
      arma::zeros<arma::vec>(data_unlabeled.num_train_ex(),1);
  }

  ~KMeansClustering() {}

  // Finds closest centroid to each training example.
  int FindClosestCentroids(const DataUnlabeled &data_unlabeled);

  // Updates centroids based on centroid assignments.
  int ComputeCentroids(const DataUnlabeled &data_unlabeled);
  
  // Runs full algorithm by iteratively calling FindClosestCentroids and 
  // ComputeCentroids.
  int Run(const DataUnlabeled &data_unlabeled,int max_iter);

  // Randomly initializes centroids.
  int InitCentroids(const DataUnlabeled &data_unlabeled,int num_centroids);

  inline int num_centroids() const {
    return num_centroids_;
  }

  inline arma::mat centroids() const {
    return centroids_;
  }

  inline arma::vec centroid_assignments() const {
    return centroid_assignments_;
  }

  inline int set_num_centroids(int num_centroids_arg) {
    num_centroids_ = num_centroids_arg;

    return 0;
  }

  inline int set_centroids(arma::mat centroids_arg) {
    centroids_ = centroids_arg;

    return 0;
  }

  inline int set_centroid_assignments(arma::vec centroid_assignments_arg) {
    centroid_assignments_ = centroid_assignments_arg;

    return 0;
  }

 private:
  // Number of centroids.
  int num_centroids_;

  // Centroids.
  arma::mat centroids_;

  // Centroid assignments.
  arma::vec centroid_assignments_;

  DISALLOW_COPY_AND_ASSIGN(KMeansClustering);
};

#endif	// MACHINE_LEARNING_PROGRAMMING_EXERCISE_7_EX7_KMEANS_CLUSTERING_H_
