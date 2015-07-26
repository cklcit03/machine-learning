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

// Declares function that computes normal equations.

#ifndef MACHINE_LEARNING_PROGRAMMING_EXERCISE_1_EX1_MULTI_NORMAL_EQN_H_
#define MACHINE_LEARNING_PROGRAMMING_EXERCISE_1_EX1_MULTI_NORMAL_EQN_H_

#include "armadillo"

#include "data.h"

arma::vec NormalEqn(const DataNormalized &data);

#endif  // MACHINE_LEARNING_PROGRAMMING_EXERCISE_1_EX1_MULTI_NORMAL_EQN_H
