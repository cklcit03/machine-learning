// Machine Learning
// Programming Exercise 1: Linear Regression
// Problem: Predict profits for a food truck given data for profits/populations of various cities
// Linear regression with one variable

#include "ex1.h"

int main(void)
{
	rowvec popVec1 = ones<rowvec>(2),popVec2 = ones<rowvec>(2);
	vec thetaFinal;
	vec thetaVec = randu<vec>(2,1);
	std::string dataFileName = "../../foodTruckData.txt";
	double alpha = 0.01,initCost = 0.0,predProfit1 = 0.0,predProfit1Scaled = 0.0,predProfit2 = 0.0,predProfit2Scaled = 0.0;
	int returnCode;
	int iterations = 1500;
	thetaVec.zeros(2,1);
	Algorithm gradDes(alpha,iterations,thetaVec);
	Data foodTruckData(dataFileName);

    // Compute initial cost
    initCost = computeCost(foodTruckData,gradDes);
    printf("initial cost = %f\n",initCost);

    // Run gradient descent
    returnCode = gradientDescent(foodTruckData,gradDes);
	thetaFinal = gradDes.getTheta();
    printf("gradient descent returns theta = \n");
	thetaFinal.print();
	printf("\n");

    // Predict profit for population size of 35000
	popVec1(1) = 3.5;
    predProfit1 = as_scalar(popVec1*thetaFinal);
    predProfit1Scaled = 10000*predProfit1;
    printf("predicted profit (in dollars) for population size of 35000 = %f\n",predProfit1Scaled);

    // Predict profit for population size of 70000
    popVec2(1) = 7.0;
    predProfit2 = as_scalar(popVec2*thetaFinal);
    predProfit2Scaled = 10000*predProfit2;
    printf("predicted profit (in dollars) for population size of 70000 = %f\n",predProfit2Scaled);

	return 0;
}
