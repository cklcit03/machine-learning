#include <string>

#include "algorithm.h"
#include "data.h"

// Compute cost function J(\theta)
double computeCost(const Data &data,const Algorithm &algorithm);

// Run gradient descent
int gradientDescent(const Data &data,Algorithm &algorithm);
