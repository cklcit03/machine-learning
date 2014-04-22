// Machine Learning
// Programming Exercise 1: Linear Regression
// Problem: Predict housing prices given sizes/bedrooms of various houses
// Linear regression with multiple variables

#include "ex1_multi.h"

int main(void)
{
	rowvec houseVec1 = ones<rowvec>(2),houseVec2 = ones<rowvec>(3);
	rowvec houseVec1Normalized,houseVec1NormalizedAug;
	vec thetaFinal,thetaNormal;
	vec thetaVec = randu<vec>(3,1);
	std::string dataFileName = "../../housingData.txt";
	double alpha = 0.1,predPrice1 = 0.0,predPrice2 = 0.0;
	int returnCode;
	int iterations = 400;
	thetaVec.zeros(3,1);
	Algorithm gradDes(alpha,iterations,thetaVec);
	DataNormalized housingData(dataFileName);

    // Perform feature normalization
	returnCode = featureNormalize(housingData);

    // Run gradient descent
    returnCode = gradientDescentMulti(housingData,gradDes);
	thetaFinal = gradDes.getTheta();
    printf("gradient descent returns theta = \n");
	thetaFinal.print();
	printf("\n");

    // Predict price for a 1650 square-foot house with 3 bedrooms
	houseVec1(0) = 1650;
	houseVec1(1) = 3;
	houseVec1Normalized = (houseVec1-housingData.getMuVec().t())/housingData.getSigmaVec().t();
	houseVec1NormalizedAug = join_horiz(ones<vec>(1),houseVec1Normalized);
    predPrice1 = as_scalar(houseVec1NormalizedAug*thetaFinal);
    printf("predicted price (in dollars) for 1650 square-foot house with 3 bedrooms = %f\n",predPrice1);

	// Solve normal equations
	thetaNormal = normalEqn(housingData);
	printf("normal equations return theta = \n");
	thetaNormal.print();
	printf("\n");

    // Use normal equations to predict price for a 1650 square-foot house with 3 bedrooms
    houseVec2(1) = 1650;
	houseVec2(2) = 3;
    predPrice2 = as_scalar(houseVec2*thetaNormal);
    printf("predicted price (in dollars) for 1650 square-foot house with 3 bedrooms using normal equations = %f\n",predPrice2);

	return 0;
}
