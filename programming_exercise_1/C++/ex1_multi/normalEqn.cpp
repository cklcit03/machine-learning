#include "ex1_multi.h"

// Compute normal equations
vec normalEqn(const DataNormalized &data)
{
	mat X = data.getTrainingFeatures();
	vec thetaNormal;
	vec y = data.getTrainingLabels();

	thetaNormal = pinv(X.t()*X)*X.t()*y;
    
	return thetaNormal;
}