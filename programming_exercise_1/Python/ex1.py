# Machine Learning
# Programming Exercise 1: Linear Regression
# Problem: Predict profits for a food truck given data for profits/populations of various cities
# Linear regression with one variable
import numpy as np

# Compute cost function J(\theta)
def computeCost(X,y,theta):
    "Compute cost function J(\theta)"
    numTrainEx = y.shape[0]
    diffVec = np.subtract(np.dot(X,theta),y)
    diffVecSq = np.multiply(diffVec,diffVec)
    if (numTrainEx > 0):
        jTheta = (np.sum(diffVecSq,axis=0))/(2*numTrainEx)
    else:
        jTheta = 0
    return(np.asscalar(jTheta))

# Run gradient descent
def gradientDescent(X,y,theta,alpha,numiters):
    "Run gradient descent"
    numTrainEx = y.shape[0]
    jThetaArray = np.zeros((numiters,1))
    if (numTrainEx > 0):
        if (numiters >= 1):
            for thetaIndex in range(0,numiters):
                diffVec = np.subtract(np.dot(X,theta),y)
                diffVecTimesX = np.c_[np.multiply(diffVec,np.reshape(X[:,0],(numTrainEx,1))),np.multiply(diffVec,np.reshape(X[:,1],(numTrainEx,1)))]
                thetaNew = np.subtract(theta,alpha*(1/numTrainEx)*np.reshape(np.sum(diffVecTimesX,axis=0),(2,1)))
                jThetaArray[thetaIndex] = computeCost(X,y,thetaNew)
                theta = thetaNew
        else:
            theta = np.zeros((2,1))
    else:
        theta = np.zeros((2,1))
    return(theta)

# Main function
def main():
    "Main function"
    foodTruckData = np.genfromtxt("../foodTruckData.txt",delimiter=",")
    numTrainEx = foodTruckData.shape[0]
    onesVec = np.ones((numTrainEx,1))
    xMat = np.c_[onesVec,foodTruckData[:,0]]
    thetaVec = np.zeros((2,1))
    yVec = np.reshape(foodTruckData[:,1],(numTrainEx,1))

    # Compute initial cost
    initCost = computeCost(xMat,yVec,thetaVec)
    print("initial cost = %f" % initCost)

    # Run gradient descent
    iterations = 1500
    alpha = 0.01
    thetaFinal = gradientDescent(xMat,yVec,thetaVec,alpha,iterations)
    print("gradient descent returns theta = %s" % np.array_str(thetaFinal))

    # Predict profit for population size of 35000
    predProfit1 = np.asscalar(np.dot(np.array([1,3.5]),thetaFinal))
    predProfit1Scaled = 10000*predProfit1
    print("predicted profit (in dollars) for population size of 35000 = %f" % predProfit1Scaled)

    # Predict profit for population size of 70000
    predProfit2 = np.asscalar(np.dot(np.array([1,7.0]),thetaFinal))
    predProfit2Scaled = 10000*predProfit2
    print("predicted profit (in dollars) for population size of 70000 = %f" % predProfit2Scaled)

# Call main function
if __name__ == "__main__":
    main()
