#include <string>

#include "algorithm.h"
#include "data.h"

// Perform feature normalization
int featureNormalize(DataNormalized &data);

// Compute cost function J(\theta)
double computeCostMulti(const DataNormalized &data,const Algorithm &algorithm);

// Run gradient descent
int gradientDescentMulti(const DataNormalized &data,Algorithm &algorithm);

// Compute normal equations
vec normalEqn(const DataNormalized &data);
