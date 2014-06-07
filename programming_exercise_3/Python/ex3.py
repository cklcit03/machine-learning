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
# Programming Exercise 3: (Multi-class) Logistic Regression and Neural Networks
# Problem: Predict label for a handwritten digit given data for 
# pixel values of various handwritten digits
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

# Display 2D data in a grid
def displayData(X):
    "Display 2D data in a grid"
    exampleWidth = (np.round(np.sqrt(X.shape[1]))).astype(int)
    numRows = X.shape[0]
    numCols = X.shape[1]
    exampleHeight = (numCols/exampleWidth)
    displayRows = (np.floor(np.sqrt(numRows))).astype(int)
    displayCols = (np.ceil(numRows/displayRows)).astype(int)
    pad = 1
    displayArray = (-1)*np.ones((pad+displayRows*(exampleHeight+pad),pad+displayCols*(exampleWidth+pad)))
    currEx = 1
    for rowIndex in range(1,displayRows+1):
      for colIndex in range(1,displayCols+1):
        if (currEx > numRows):
          break
        maxVal = np.amax(np.absolute(X[currEx-1,:]))
        minRowIdx = pad+(rowIndex-1)*(exampleHeight+pad)
        maxRowIdx = pad+(rowIndex-1)*(exampleHeight+pad)+exampleHeight
        minColIdx = pad+(colIndex-1)*(exampleWidth+pad)
        maxColIdx = pad+(colIndex-1)*(exampleWidth+pad)+exampleWidth
        xReshape = np.reshape(X[currEx-1,],(exampleHeight,exampleWidth))
        displayArray[minRowIdx:maxRowIdx,minColIdx:maxColIdx] = (1/maxVal)*np.fliplr(np.rot90(xReshape,3))
        currEx = currEx+1
      if (currEx > numRows):
        break
    plt.imshow(displayArray,cmap=cm.Greys_r)
    plt.axis('off')

    return None

# Compute sigmoid function
def computeSigmoid(z):
    "Compute sigmoid function"
    sigmoidZ = 1/(1+np.exp(-z))

    return(sigmoidZ)

# Compute regularized cost function J(\theta)
def computeCost(theta,X,y,numTrainEx,lamb):
    "Compute regularized cost function J(\theta)"
    numFeatures = X.shape[1]
    theta = np.reshape(theta,(numFeatures,1),order='F')
    hTheta = computeSigmoid(np.dot(X,theta))
    thetaSquared = np.power(theta,2)
    jTheta = (np.sum(np.subtract(np.multiply(-y,np.log(hTheta)),np.multiply((1-y),np.log(1-hTheta))),axis=0))/numTrainEx
    jThetaReg = jTheta+(lamb/(2*numTrainEx))*np.sum(thetaSquared,axis=0)-thetaSquared[0]

    return(jThetaReg)

# Compute gradient of regularized cost function J(\theta)
def computeGradient(theta,X,y,numTrainEx,lamb):
    "Compute gradient of regularized cost function J(\theta)"
    numFeatures = X.shape[1]
    theta = np.reshape(theta,(numFeatures,1),order='F')
    hTheta = computeSigmoid(np.dot(X,theta))
    gradArray = np.zeros((numFeatures,1))
    gradArrayReg = np.zeros((numFeatures,1))
    if (numFeatures == 0):
        raise InsufficientFeatures('numFeatures = 0')
    for gradIndex in range(0,numFeatures):
        gradTerm = np.multiply(np.reshape(X[:,gradIndex],(numTrainEx,1)),np.subtract(hTheta,y))
        gradArray[gradIndex] = (np.sum(gradTerm,axis=0))/numTrainEx
        gradArrayReg[gradIndex] = gradArray[gradIndex]+(lamb/numTrainEx)*theta[gradIndex]

    gradArrayReg[0] = gradArrayReg[0]-(lamb/numTrainEx)*theta[0]
    gradArrayRegFlat = np.ndarray.flatten(gradArrayReg)
    return(gradArrayRegFlat)

# Train multiple logistic regression classifiers
def oneVsAll(X,y,numLabels,lamb):
    "Train multiple logistic regression classifiers"
    numTrainEx = X.shape[0]
    numFeatures = X.shape[1]
    allTheta = np.zeros((numLabels,numFeatures+1))
    onesVec = np.ones((numTrainEx,1))
    augX = np.c_[onesVec,X]
    for labelIndex in range(0,numLabels):
      thetaVec = np.zeros((numFeatures+1,1))
      thetaVecFlat = np.ndarray.flatten(thetaVec)
      yArg = (np.equal(y,(labelIndex+1)*np.ones((numTrainEx,1)))).astype(int)
      fminNcgOut = fmin_ncg(computeCost,thetaVecFlat,fprime=computeGradient,args=(augX,yArg,numTrainEx,lamb),avextol=1e-10,epsilon=1e-10,maxiter=400,full_output=1)
      thetaOpt = np.reshape(fminNcgOut[0],(1,numFeatures+1),order='F')
      allTheta[labelIndex,:] = thetaOpt

    return(allTheta)

# Perform label prediction on training data
def predictOneVsAll(X,allTheta):
    "Perform label prediction on training data"
    numTrainEx = X.shape[0]
    onesVec = np.ones((numTrainEx,1))
    augX = np.c_[onesVec,X]
    sigmoidArray = computeSigmoid(np.dot(augX,np.transpose(allTheta)))
    p = np.argmax(sigmoidArray,axis=1)
    for exampleIndex in range(0,numTrainEx):
        p[exampleIndex] = p[exampleIndex]+1

    return(p)

# Main function
def main():
    "Main function"
    print("Loading and Visualizing Data ...")
    digitData = np.genfromtxt("../digitData.txt",delimiter=",")
    numTrainEx = digitData.shape[0]
    numFeatures = digitData.shape[1]-1
    xMat = digitData[:,0:numFeatures]
    yVec = digitData[:,numFeatures:(numFeatures+1)]

    # Randomly select 100 data points to display
    randIndices = np.random.permutation(numTrainEx)
    xMatSel = xMat[randIndices[0],:]
    for randIndex in range(1,100):
        xMatSel = np.vstack([xMatSel,xMat[randIndices[randIndex],:]])
    returnCode = displayData(xMatSel)
    plt.show()
    input("Program paused. Press enter to continue.")

    # Train one logistic regression classifier for each digit
    print("\n")
    print("Training One-vs-All Logistic Regression...")
    lamb = 0.1
    numLabels = 10
    allTheta = oneVsAll(xMat,yVec,numLabels,lamb)
    input("Program paused. Press enter to continue.")

    # Perform one-versus-all classification using logistic regression
    trainingPredict = predictOneVsAll(xMat,allTheta)
    numTrainMatch = 0
    for trainIndex in range(0,numTrainEx):
        if (trainingPredict[trainIndex] == yVec[trainIndex]):
            numTrainMatch += 1
    print("\n")
    print("Training Set Accuracy: %.6f" % (100*numTrainMatch/numTrainEx))

# Call main function
if __name__ == "__main__":
    main()
