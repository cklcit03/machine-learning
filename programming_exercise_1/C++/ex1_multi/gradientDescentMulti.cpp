#include "ex1_multi.h"

// Run gradient descent
int gradientDescentMulti(const DataNormalized &data,Algorithm &algorithm)
{
	mat diffVecTimesX;
	mat X = data.getTrainingFeaturesNormalized();
	vec diffVec,thetaNew;
	vec theta = algorithm.getTheta();
	vec y = data.getTrainingLabels();
	int numFeatures = data.getNumFeatures();
	int numIters = algorithm.getNumIters();
	int numTrainEx = data.getNumTrainEx();
	int featureIndex = 0,thetaIndex = 0;
    double *jThetaArray = new double[numIters];
	double alpha = algorithm.getAlpha();
    
	if (numTrainEx > 0)
	{
		if (numFeatures >= 2)
		{
			if (numIters >= 1)
			{
				for(thetaIndex=0;thetaIndex<numIters;thetaIndex++)
				{
					diffVec = X*theta-y;
					diffVecTimesX = diffVec % X.col(0);
					for(featureIndex=1;featureIndex<=numFeatures;featureIndex++)
						diffVecTimesX = join_rows(diffVecTimesX,diffVec % X.col(featureIndex));
					thetaNew = theta-alpha*(1/(float)numTrainEx)*(sum(diffVecTimesX)).t();
					algorithm.setTheta(thetaNew);
					jThetaArray[thetaIndex] = computeCostMulti(data,algorithm);
					theta = thetaNew;
				}
			}
			else
			{
				printf("Insufficient iterations!\n");
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			printf("Insufficient features!\n");
			exit(EXIT_FAILURE);
		}
	}
    else
	{
        printf("Insufficient training examples!\n");
		exit(EXIT_FAILURE);
	}

	delete [] jThetaArray;

	return 0;
}
