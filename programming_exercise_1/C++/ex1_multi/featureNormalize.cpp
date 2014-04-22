#include "ex1_multi.h"

// Perform feature normalization
int featureNormalize(DataNormalized &data)
{
	int numFeatures = data.getNumFeatures();
    int numTrainEx = data.getNumTrainEx();
	int index;
	mat X = (data.getTrainingFeatures()).cols(1,numFeatures);
	mat XNormalized = zeros<mat>(numTrainEx,numFeatures);
	mat XNormalizedAug;
	vec muVec,sigmaVec;

	muVec = mean(X.cols(0,numFeatures-1)).t();
	data.setMuVec(muVec);
    if (numFeatures >= 1)
	{
		sigmaVec = stddev(X.cols(0,numFeatures-1)).t();
		data.setSigmaVec(sigmaVec);
		if (numTrainEx >= 1)
		{
			for(index=0;index<numTrainEx;index++)
				XNormalized.row(index) = (X.row(index)-muVec.t())/sigmaVec.t();
			XNormalizedAug = join_horiz(ones<vec>(numTrainEx),XNormalized);
			data.setTrainingFeaturesNormalized(XNormalizedAug);
		}
		else
		{
			printf("Insufficient training examples!\n");
			exit(EXIT_FAILURE);
		}
	}
    else
	{
        printf("Insufficient features!\n");
		exit(EXIT_FAILURE);
	}
    
	return 0;
}