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

// Declares function that loads a list of movies.

#include "load_movie_list.h"

// The file that this function reads consists of a list of movies.
// Each line in this file has the format: "movieID movie".
// This function should ignore the "movieID" field.
int LoadMovieList(std::vector<std::string> *movie_list)
{
  std::string curr_line;
  std::ifstream movie_ids_file("../../movie_ids.txt");
  assert(movie_ids_file.is_open());
  while (std::getline(movie_ids_file,curr_line)) {

    // Splits current line into tokens and discards first token.
    // The second token corresponds to the movie.
    char *first_token = strtok((char *)curr_line.c_str()," ");
    char *curr_token = strtok(NULL," ");
    std::string movie(curr_token);
    curr_token = strtok(NULL," ");
    while (curr_token != NULL) {
      movie += " ";
      movie += std::string(curr_token);
      curr_token = strtok(NULL," ");
    }
    movie_list->push_back(movie);
  }
  movie_ids_file.close();

  return 0;
}
