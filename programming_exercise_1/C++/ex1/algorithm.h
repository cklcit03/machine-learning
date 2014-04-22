#include "armadillo"

using namespace arma;

class Algorithm
{
	vec theta;
	double alpha;
	int numIters;
public:
	Algorithm(double alphaArg,int numItersArg,vec thetaArg)
	{
		alpha = alphaArg;
		numIters = numItersArg;
		theta = thetaArg;
	}
	~Algorithm()
	{
	}

	vec getTheta() const
	{
		return theta;
	}
	double getAlpha() const
	{
		return alpha;
	}
	int getNumIters() const
	{
		return numIters;
	}
	int setTheta(vec thetaArg)
	{
		theta = thetaArg;
		
		return 0;
	}
};
