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

#ifndef LOAD_MOVIE_LIST_H_
#define LOAD_MOVIE_LIST_H_

#include <assert.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int LoadMovieList(std::vector<std::string> *movie_list);

#endif  // LOAD_MOVIE_LIST_H
