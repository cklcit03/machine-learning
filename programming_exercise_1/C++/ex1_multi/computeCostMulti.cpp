#include "ex1_multi.h"

// Compute cost function J(\theta)
double computeCostMulti(const DataNormalized &data,const Algorithm &algorithm)
{
	mat X = data.getTrainingFeatures();
	vec diffVec,diffVecSq;
	vec theta = algorithm.getTheta();
	vec y = data.getTrainingLabels();
	double jTheta = 0.0;
    int numTrainEx = data.getNumTrainEx();

	diffVec = X*theta-y;
    diffVecSq = diffVec % diffVec;
    if (numTrainEx > 0)
		jTheta = as_scalar(sum(diffVecSq))/(2.0*numTrainEx);
    else
	{
        printf("Insufficient training examples!\n");
		exit(EXIT_FAILURE);
	}
    
	return jTheta;
}