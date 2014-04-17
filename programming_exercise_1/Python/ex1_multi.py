# Machine Learning
# Programming Exercise 1: Linear Regression
# Problem: Predict housing prices given sizes/bedrooms of various houses
# Linear regression with multiple variables
import numpy as np

# Perform feature normalization
def featureNormalize(X):
    "Perform feature normalization"
    numTrainEx = X.shape[0]
    numFeatures = X.shape[1]
    xNormalized = np.zeros((numTrainEx,numFeatures))
    muVec = np.mean(X,axis=0)
    if (numFeatures >= 1):
        sigmaVec = np.std(X,axis=0)
        if (numTrainEx >= 1):
            for index in range(0,numTrainEx):
                xNormalized[index] = np.divide(np.subtract(X[index,:],muVec),sigmaVec)
        else:
            xNormalized = 0
            muVec = np.zeros((1,2))
            sigmaVec = np.ones((1,2))
    else:
        xNormalized = np.zeros((numTrainEx,0))
        muVec = np.zeros((1,2))
        sigmaVec = np.ones((1,2))
    returnList = {'xNormalized': xNormalized,'muVec': muVec,'sigmaVec': sigmaVec}
    return(returnList)

# Compute cost function J(\theta)
def computeCostMulti(X,y,theta):
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
def gradientDescentMulti(X,y,theta,alpha,numiters):
    "Run gradient descent"
    numTrainEx = y.shape[0]
    numFeatures = X.shape[1]
    jThetaArray = np.zeros((numiters,1))
    if (numTrainEx > 0):
        if (numFeatures >= 2):
            if (numiters >= 1):
                for thetaIndex in range(0,numiters):
                    diffVec = np.subtract(np.dot(X,theta),y)
                    diffVecTimesX = np.multiply(diffVec,np.reshape(X[:,0],(numTrainEx,1)))
                    for featureIndex in range(1,numFeatures):
                        diffVecTimesX = np.c_[diffVecTimesX,np.multiply(diffVec,np.reshape(X[:,featureIndex],(numTrainEx,1)))]
                    thetaNew = np.subtract(theta,alpha*(1/numTrainEx)*np.reshape(np.sum(diffVecTimesX,axis=0),(numFeatures,1)))
                    jThetaArray[thetaIndex] = computeCostMulti(X,y,thetaNew)
                    theta = thetaNew
            else:
                theta = np.zeros((3,1))
        else:
            theta = np.zeros((3,1))
    else:
        theta = np.zeros((3,1))
    return(theta)

# Compute normal equations
def normalEqn(X,y):
    "Compute normal equations"
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)),np.transpose(X)),y)
    return(theta)

# Main function
def main():
    "Main function"
    housingData = np.genfromtxt("../housingData.txt",delimiter=",")
    xMat = np.c_[housingData[:,0],housingData[:,1]]

    # Perform feature normalization
    featureNormalizeList = featureNormalize(xMat)
    xMatNormalized = featureNormalizeList['xNormalized']
    muVec = featureNormalizeList['muVec']
    sigmaVec = featureNormalizeList['sigmaVec']
    numTrainEx = housingData.shape[0]
    onesVec = np.ones((numTrainEx,1))
    xMatAug = np.c_[onesVec,xMat]
    xMatNormalizedAug = np.c_[onesVec,xMatNormalized]
    yVec = np.reshape(housingData[:,2],(numTrainEx,1))
    thetaVec = np.zeros((3,1))
    iterations = 400
    alpha = 0.1

    # Run gradient descent
    thetaFinal = gradientDescentMulti(xMatNormalizedAug,yVec,thetaVec,alpha,iterations)
    print("gradient descent returns theta = %s" % np.array_str(thetaFinal))

    # Predict price for a 1650 square-foot house with 3 bedrooms
    xMatNormalized1 = np.reshape(np.divide(np.subtract(np.array([1650,3]),muVec),sigmaVec),(1,2))
    xMatNormalized1Aug = np.c_[np.ones((1,1)),xMatNormalized1]
    predPrice1 = np.asscalar(np.dot(xMatNormalized1Aug,thetaFinal))
    print("predicted price (in dollars) for 1650 square-foot house with 3 bedrooms = %f" % predPrice1)

    # Solve normal equations
    thetaNormal = normalEqn(xMatAug,yVec)
    print("normal equations return theta = %s" % np.array_str(thetaNormal))

    # Use normal equations to predict price for a 1650 square-foot house with 3 bedrooms
    xMat2 = np.array([1,1650,3])
    predPrice2 = np.asscalar(np.dot(xMat2,thetaNormal))
    print("predicted price (in dollars) for 1650 square-foot house with 3 bedrooms using normal equations = %f" % predPrice2)

# Call main function
if __name__ == "__main__":
    main()
