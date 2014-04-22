#include <string>

#include "armadillo"

using namespace arma;

class Data
{
	mat trainingFeatures;
	vec trainingLabels;
	int numFeatures;
	int numTrainEx;
public:
	Data(std::string fileNameArg)
	{
		mat trainingData;
		trainingData.load(fileNameArg,csv_ascii);
		numTrainEx = trainingData.n_rows;
		numFeatures = trainingData.n_cols-1;
		trainingFeatures = join_horiz(ones<vec>(numTrainEx),trainingData.cols(0,numFeatures-1));
		trainingLabels = trainingData.col(numFeatures);
	}
	~Data()
	{
	}

	mat loadedData;
	mat getTrainingFeatures() const
	{
		return trainingFeatures;
	}
	vec getTrainingLabels() const
	{
		return trainingLabels;
	}
	int getNumFeatures() const
	{
		return numFeatures;
	}
	int getNumTrainEx() const
	{
		return numTrainEx;
	}
};

class DataNormalized: public Data
{
	mat trainingFeaturesNormalized;
	vec muVec;
	vec sigmaVec;
public:
	DataNormalized(std::string fileNameArg) : Data(fileNameArg)
	{
	}
	~DataNormalized()
	{
	}

	mat getTrainingFeaturesNormalized() const
	{
		return trainingFeaturesNormalized;
	}
	mat getMuVec() const
	{
		return muVec;
	}
	mat getSigmaVec() const
	{
		return sigmaVec;
	}
	int setTrainingFeaturesNormalized(mat trainingFeaturesNormalizedArg)
	{
		trainingFeaturesNormalized = trainingFeaturesNormalizedArg;

		return 0;
	}
	int setMuVec(vec muVecArg)
	{
		muVec = muVecArg;

		return 0;
	}
	int setSigmaVec(vec sigmaVecArg)
	{
		sigmaVec = sigmaVecArg;

		return 0;
	}
};