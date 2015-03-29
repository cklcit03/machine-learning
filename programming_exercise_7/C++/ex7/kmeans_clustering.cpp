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

#include "kmeans_clustering.h"

// Finds closest centroid to each training example.
int KMeansClustering::FindClosestCentroids(const DataUnlabeled \
  &data_unlabeled) {
  const int kNumTrainEx = data_unlabeled.num_train_ex();
  assert(kNumTrainEx > 0);
  for(int ex_index=0;ex_index<kNumTrainEx;ex_index++)
  {
    centroid_assignments_.at(ex_index) = 1;

    // Considers Euclidean distance.
    double min_distance = \
      arma::norm(data_unlabeled.training_features().row(ex_index)-\
      centroids_.row(0),2);
    for(int cent_index=1;cent_index<(num_centroids_);cent_index++)
    {
      double tmp_distance = \
        arma::norm(data_unlabeled.training_features().row(ex_index)-\
        centroids_.row(cent_index),2);
      if (tmp_distance < min_distance) {
        min_distance = tmp_distance;
        centroid_assignments_.at(ex_index) = cent_index+1;
      }
    }
  }

  return 0;
}

// Updates centroids based on centroid assignments.
int KMeansClustering::ComputeCentroids(const DataUnlabeled \
  &data_unlabeled) {
  const int kNumTrainEx = data_unlabeled.num_train_ex();
  assert(kNumTrainEx > 0);
  for(int cent_index=0;cent_index<(num_centroids_);cent_index++)
  {
    arma::mat is_centroid_idx = arma::zeros<arma::mat>(kNumTrainEx,1);
    for(int ex_index=0;ex_index<kNumTrainEx;ex_index++)
    {
      is_centroid_idx.at(ex_index,0) = (centroid_assignments_.at(ex_index,0) \
        == cent_index+1) ? 1 : 0;
    }
    arma::mat sum_centroid_points = \
      is_centroid_idx.t()*data_unlabeled.training_features();
    centroids_.row(cent_index) = \
      sum_centroid_points/arma::as_scalar(arma::sum(is_centroid_idx));
  }

  return 0;
}

// Runs full algorithm by iteratively calling FindClosestCentroids and 
// ComputeCentroids.
int KMeansClustering::Run(const DataUnlabeled &data_unlabeled,int max_iter) {
  assert(max_iter > 0);
  for(int iter_index=0;iter_index<(max_iter);iter_index++)
  {
    printf("K-Means iteration %d/%d...\n",iter_index+1,max_iter);
    const int kReturnCode = this->FindClosestCentroids(data_unlabeled);
    const int kReturnCode2 = this->ComputeCentroids(data_unlabeled);
  }

  return 0;
}

// Randomly permutes training data and selects first num_centroids permuted 
// examples.
int KMeansClustering::InitCentroids(const DataUnlabeled &data_unlabeled,\
  int num_centroids) {
  std::vector<int> vector_indices;
  for(int index=0;index<data_unlabeled.num_train_ex();index++)
  {
    vector_indices.push_back(index);
  }
  std::random_shuffle(vector_indices.begin(),vector_indices.end());
  centroids_ = \
    arma::zeros<arma::mat>(num_centroids,data_unlabeled.num_features());
  for(int centroid_idx=0;centroid_idx<num_centroids;centroid_idx++)
  {
    centroids_.row(centroid_idx) = \
      data_unlabeled.training_features().row(vector_indices.at(centroid_idx));
  }

  return 0;
}
