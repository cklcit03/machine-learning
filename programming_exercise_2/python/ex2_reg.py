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
# Programming Exercise 2: Logistic Regression
# Problem: Predict chances of acceptance for a microchip given data for 
# acceptance decisions and test scores of various microchips
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_ncg

class InsufficientFeatures(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class InsufficientTrainingExamples(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# Plot data
def plotData(x,y):
    "Plot data"
    positiveIndices = np.where(y == 1)
    negativeIndices = np.where(y == 0)
    pos = plt.scatter(x[positiveIndices,0],x[positiveIndices,1],s=80,marker='+',color='k')
    plt.hold(True)
    neg = plt.scatter(x[negativeIndices,0],x[negativeIndices,1],s=80,marker='s',color='y')
    plt.legend((pos,neg),('y = 1','y = 0'),loc='lower right')
    plt.hold(False)
    plt.ylabel('Microchip Test 2',fontsize=18)
    plt.xlabel('Microchip Test 1',fontsize=18)

    return None

# Plot decision boundary
def plotDecisionBoundary(x,y,theta):
    "Plot decision boundary"
    plotData(np.c_[x[:,0],x[:,1]],y)
    plt.hold(True)
    theta0Vals = np.linspace(-1,1.5,num=50)
    theta1Vals = np.linspace(-1,1.5,num=50)
    jVals = np.zeros((theta0Vals.shape[0],theta1Vals.shape[0]))
    for theta0Index in range(0,theta0Vals.shape[0]):
        for theta1Index in range(0,theta1Vals.shape[0]):
            jVals[theta0Index,theta1Index] = np.dot(mapFeature(theta0Vals[theta0Index],theta1Vals[theta1Index]),theta)
    jValsTrans = np.transpose(jVals)
    theta0ValsX,theta1ValsY = np.meshgrid(theta0Vals,theta1Vals)
    jValsReshape = jValsTrans.reshape(theta0ValsX.shape)
    plt.contour(theta0Vals,theta1Vals,jValsReshape,1)
    plt.hold(False)

    return None

# Add polynomial features to training data
def mapFeature(X1,X2):
    "Add polynomial features to training data"
    degree = 6
    numTrainEx = np.c_[X1,X2].shape[0]
    augXMat = np.ones((numTrainEx,1))
    for degIndex1 in range(1,degree+1):
        for degIndex2 in range(0,degIndex1+1):
            augXMat = np.c_[augXMat,np.multiply(np.power(X1,(degIndex1-degIndex2)),np.power(X2,degIndex2))]

    return(augXMat)

# Compute sigmoid function
def computeSigmoid(z):
    "Compute sigmoid function"
    sigmoidZ = 1/(1+np.exp(-z))

    return(sigmoidZ)

# Compute regularized cost function J(\theta)
def computeCost(theta,X,y,numTrainEx,lamb):
    "Compute regularized cost function J(\theta)"
    numFeatures = X.shape[1]
    if (numFeatures == 0):
        raise InsufficientFeatures('numFeatures = 0')
    theta = np.reshape(theta,(numFeatures,1),order='F')
    hTheta = computeSigmoid(np.dot(X,theta))
    thetaSquared = np.power(theta,2)
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    jTheta = (np.sum(np.subtract(np.multiply(-y,np.log(hTheta)),np.multiply((1-y),np.log(1-hTheta))),axis=0))/numTrainEx
    jThetaReg = jTheta+(lamb/(2*numTrainEx))*np.sum(thetaSquared,axis=0)-thetaSquared[0]

    return(jThetaReg)

# Compute gradient of regularized cost function J(\theta)
def computeGradient(theta,X,y,numTrainEx,lamb):
    "Compute gradient of regularized cost function J(\theta)"
    numFeatures = X.shape[1]
    if (numFeatures == 0):
        raise InsufficientFeatures('numFeatures = 0')
    theta = np.reshape(theta,(numFeatures,1),order='F')
    hTheta = computeSigmoid(np.dot(X,theta))
    gradArray = np.zeros((numFeatures,1))
    gradArrayReg = np.zeros((numFeatures,1))
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    for gradIndex in range(0,numFeatures):
        gradTerm = np.multiply(np.reshape(X[:,gradIndex],(numTrainEx,1)),np.subtract(hTheta,y))
        gradArray[gradIndex] = (np.sum(gradTerm,axis=0))/numTrainEx
        gradArrayReg[gradIndex] = gradArray[gradIndex]+(lamb/numTrainEx)*theta[gradIndex]

    gradArrayReg[0] = gradArrayReg[0]-(lamb/numTrainEx)*theta[0]
    gradArrayRegFlat = np.ndarray.flatten(gradArrayReg)
    return(gradArrayRegFlat)

# Aggregate computed cost and gradient
def computeCostGradList(X,y,theta,lamb):
    "Aggregate computed cost and gradient"
    numFeatures = X.shape[1]
    numTrainEx = y.shape[0]
    jThetaReg = computeCost(theta,X,y,numTrainEx,lamb)
    gradArrayRegFlat = computeGradient(theta,X,y,numTrainEx,lamb)
    gradArrayReg = np.reshape(gradArrayRegFlat,(numFeatures,1),order='F')
    returnList = {'jThetaReg': jThetaReg,'gradArrayReg': gradArrayReg}

    return(returnList)

# Perform label prediction on training data
def labelPrediction(X,theta):
    "Perform label prediction on training data"
    numTrainEx = X.shape[0]
    sigmoidArray = computeSigmoid(np.dot(X,theta))
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    p = np.zeros((numTrainEx,1))
    for trainIndex in range(0,numTrainEx):
        if (sigmoidArray[trainIndex] >= 0.5):
            p[trainIndex] = 1
        else:
            p[trainIndex] = 0

    return(p)

# Main function
def main():
    "Main function"
    microChipData = np.genfromtxt("../microChipData.txt",delimiter=",")
    returnCode = plotData(np.c_[microChipData[:,0],microChipData[:,1]],microChipData[:,2])
    plt.show()
    numTrainEx = microChipData.shape[0]
    numFeatures = microChipData.shape[1]-1
    onesVec = np.ones((numTrainEx,1))
    xMat = np.c_[microChipData[:,0],microChipData[:,1]]
    yVec = np.reshape(microChipData[:,2],(numTrainEx,1))

    # Add polynomial features to training data
    featureXMat = mapFeature(xMat[:,0],xMat[:,1])
    thetaVec = np.zeros((featureXMat.shape[1],1))

    # Compute initial cost and gradient
    lamb = 1
    initComputeCostList = computeCostGradList(featureXMat,yVec,thetaVec,lamb)
    print("Cost at initial theta (zeros): %.6f" % initComputeCostList['jThetaReg'])
    input("Program paused. Press enter to continue.")

    # Use fmin_ncg to solve for optimum theta and cost
    thetaVecFlat = np.ndarray.flatten(thetaVec)
    fminNcgOut = fmin_ncg(computeCost,thetaVecFlat,fprime=computeGradient,args=(featureXMat,yVec,numTrainEx,lamb),avextol=1e-10,epsilon=1e-10,maxiter=400,full_output=1)
    thetaOpt = np.reshape(fminNcgOut[0],(featureXMat.shape[1],1),order='F')

    # Plot decision boundary
    returnCode = plotDecisionBoundary(xMat,yVec,thetaOpt)
    plt.show()

    # Compute accuracy on training set
    trainingPredict = labelPrediction(featureXMat,thetaOpt)
    numTrainMatch = 0
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    for trainIndex in range(0,numTrainEx):
        if (trainingPredict[trainIndex] == yVec[trainIndex]):
            numTrainMatch += 1
    print("Train Accuracy: %.6f" % (100*numTrainMatch/numTrainEx))

# Call main function
if __name__ == "__main__":
    main()
