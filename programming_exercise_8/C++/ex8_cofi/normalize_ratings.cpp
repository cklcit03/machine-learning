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

// Declares function that normalizes a list of movie ratings.

#include "normalize_ratings.h"

// For each movie, ignores those users who did not rate it.
arma::vec NormalizeRatings(const DataUnlabeled &ratings_data,
  const DataUnlabeled &indicator_data)
{
  const int kNumMovies = ratings_data.num_train_ex();
  assert(kNumMovies >= 1);
  arma::vec ratings_mean = arma::zeros<arma::vec>(kNumMovies,1);
  const arma::mat kRatingsMat = ratings_data.training_features();
  const arma::mat kIndicatorMat = indicator_data.training_features();
  for(int movie_index=0; movie_index<kNumMovies; movie_index++)
  {
    ratings_mean(movie_index) = \
      arma::as_scalar(sum(kRatingsMat.row(movie_index)))/\
      arma::as_scalar(sum(kIndicatorMat.row(movie_index)));
  }

  return ratings_mean;
}
