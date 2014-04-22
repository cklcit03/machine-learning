#include "ex1.h"

// Run gradient descent
int gradientDescent(const Data &data,Algorithm &algorithm)
{
	mat diffVecTimesX;
	mat X = data.getTrainingFeatures();
	vec diffVec,thetaNew;
	vec theta = algorithm.getTheta();
	vec y = data.getTrainingLabels();
	int numIters = algorithm.getNumIters();
	int numTrainEx = data.getNumTrainEx();
	int thetaIndex = 0;
    double *jThetaArray = new double[numIters];
	double alpha = algorithm.getAlpha();
    
	if (numTrainEx > 0)
	{
        if (numIters >= 1)
		{
            for(thetaIndex=0;thetaIndex<numIters;thetaIndex++)
			{
                diffVec = X*theta-y;
                diffVecTimesX = join_rows(diffVec % X.col(0),diffVec % X.col(1));
                thetaNew = theta-alpha*(1/(float)numTrainEx)*(sum(diffVecTimesX)).t();
				algorithm.setTheta(thetaNew);
                jThetaArray[thetaIndex] = computeCost(data,algorithm);
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
        printf("Insufficient training examples!\n");
		exit(EXIT_FAILURE);
	}

	delete [] jThetaArray;

	return 0;
}
