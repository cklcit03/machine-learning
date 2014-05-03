# Copyright (C) 2014  Caleb Lo
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Machine Learning
# Programming Exercise 1: Linear Regression
# Problem: Predict housing prices given sizes/bedrooms of various houses
# Linear regression with multiple variables
import matplotlib.pyplot as plt
import numpy as np

class InsufficientFeatures(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class InsufficientIterations(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class InsufficientTrainingExamples(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# Perform feature normalization
def featureNormalize(X):
    "Perform feature normalization"
    numTrainEx = X.shape[0]
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    numFeatures = X.shape[1]
    if (numFeatures == 0):
        raise InsufficientFeatures('numFeatures = 0')
    xNormalized = np.zeros((numTrainEx,numFeatures))
    muVec = np.mean(X,axis=0)
    sigmaVec = np.std(X,axis=0)
    for index in range(0,numTrainEx):
        xNormalized[index] = np.divide(np.subtract(X[index,:],muVec),sigmaVec)
    returnList = {'xNormalized': xNormalized,'muVec': muVec,'sigmaVec': sigmaVec}

    return(returnList)

# Compute cost function J(\theta)
def computeCostMulti(X,y,theta):
    "Compute cost function J(\theta)"
    numTrainEx = y.shape[0]
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    diffVec = np.subtract(np.dot(X,theta),y)
    diffVecSq = np.multiply(diffVec,diffVec)
    jTheta = (np.sum(diffVecSq,axis=0))/(2*numTrainEx)

    return(np.asscalar(jTheta))

# Run gradient descent
def gradientDescentMulti(X,y,theta,alpha,numiters):
    "Run gradient descent"
    if (numiters <= 0):
        raise InsufficientIterations('numiters <= 0')
    numTrainEx = y.shape[0]
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    numFeatures = X.shape[1]
    if (numFeatures == 0):
        raise InsufficientFeatures('numFeatures = 0')
    jThetaArray = np.zeros((numiters,1))
    for thetaIndex in range(0,numiters):
        diffVec = np.subtract(np.dot(X,theta),y)
        diffVecTimesX = np.multiply(diffVec,np.reshape(X[:,0],(numTrainEx,1)))
        for featureIndex in range(1,numFeatures):
            diffVecTimesX = np.c_[diffVecTimesX,np.multiply(diffVec,np.reshape(X[:,featureIndex],(numTrainEx,1)))]
        thetaNew = np.subtract(theta,alpha*(1/numTrainEx)*np.reshape(np.sum(diffVecTimesX,axis=0),(numFeatures,1)))
        jThetaArray[thetaIndex] = computeCostMulti(X,y,thetaNew)
        theta = thetaNew
    returnList = {'theta': theta,'jHistory': jThetaArray}

    return(returnList)

# Compute normal equations
def normalEqn(X,y):
    "Compute normal equations"
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)),np.transpose(X)),y)

    return(theta)

# Main function
def main():
    "Main function"
    print("Loading data ...")
    housingData = np.genfromtxt("../housingData.txt",delimiter=",")
    xMat = np.c_[housingData[:,0],housingData[:,1]]
    numTrainEx = housingData.shape[0]
    yVec = np.reshape(housingData[:,2],(numTrainEx,1))
    print("First 10 examples from the dataset:")
    for trainingExIndex in range(0,10):
        print(" x = %s, y = %s" % (np.array_str(xMat[trainingExIndex,:].astype(int)),np.array_str(yVec[trainingExIndex,:].astype(int))))
    input("Program paused. Press enter to continue.")
    print("")

    # Perform feature normalization
    print("Normalizing Features ...")
    featureNormalizeList = featureNormalize(xMat)
    xMatNormalized = featureNormalizeList['xNormalized']
    muVec = featureNormalizeList['muVec']
    sigmaVec = featureNormalizeList['sigmaVec']
    onesVec = np.ones((numTrainEx,1))
    xMatAug = np.c_[onesVec,xMat]
    xMatNormalizedAug = np.c_[onesVec,xMatNormalized]
    thetaVec = np.zeros((3,1))
    iterations = 400
    alpha = 0.1

    # Run gradient descent
    print("Running gradient descent ...")
    gradientDescentMultiList = gradientDescentMulti(xMatNormalizedAug,yVec,thetaVec,alpha,iterations)
    thetaFinal = gradientDescentMultiList['theta']
    jHistory = gradientDescentMultiList['jHistory']
    plt.plot(np.arange(jHistory.shape[0]),jHistory,'b-',markersize=18)
    plt.ylabel('Cost J',fontsize=18)
    plt.xlabel('Number of iterations',fontsize=18)
    plt.show()
    print("Theta computed from gradient descent:")
    print("%s\n" % np.array_str(np.round(thetaFinal,6)))

    # Predict price for a 1650 square-foot house with 3 bedrooms
    xMatNormalized1 = np.reshape(np.divide(np.subtract(np.array([1650,3]),muVec),sigmaVec),(1,2))
    xMatNormalized1Aug = np.c_[np.ones((1,1)),xMatNormalized1]
    predPrice1 = np.asscalar(np.dot(xMatNormalized1Aug,thetaFinal))
    print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%.6f" % predPrice1)
    input("Program paused. Press enter to continue.")
    print("")

    # Solve normal equations
    print("Solving with normal equations...")
    thetaNormal = normalEqn(xMatAug,yVec)
    print("Theta computed from the normal equations:")
    print("%s\n" % np.array_str(np.round(thetaNormal,6)))

    # Use normal equations to predict price for a 1650 square-foot house with 3 bedrooms
    xMat2 = np.array([1,1650,3])
    predPrice2 = np.asscalar(np.dot(xMat2,thetaNormal))
    print("Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%.6f" % predPrice2)

# Call main function
if __name__ == "__main__":
    main()
