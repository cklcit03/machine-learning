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
# Use parameters trained by a neural network for prediction
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

# Perform label prediction on training data
def predict(Theta1,Theta2,X):
    "Perform label prediction on training data"
    numTrainEx = X.shape[0]
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    onesVec = np.ones((numTrainEx,1))
    augX = np.c_[onesVec,X]
    hiddenLayerActivation = computeSigmoid(np.dot(augX,np.transpose(Theta1)))
    onesVecMod = np.ones((hiddenLayerActivation.shape[0],1))
    hiddenLayerActivationMod = np.c_[onesVecMod,hiddenLayerActivation]
    outputLayerActivation = computeSigmoid(np.dot(hiddenLayerActivationMod,np.transpose(Theta2)))
    p = np.argmax(outputLayerActivation,axis=1)
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

    # Load two files that contain parameters trained by a neural network into R
    print("\n")
    print("Loading Saved Neural Network Parameters ...")
    theta1Mat = np.genfromtxt("../Theta1.txt",delimiter=",")
    theta2Mat = np.genfromtxt("../Theta2.txt",delimiter=",")

    # Perform one-versus-all classification using trained parameters
    trainingPredict = predict(theta1Mat,theta2Mat,xMat)
    numTrainMatch = 0
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    for trainIndex in range(0,numTrainEx):
        if (trainingPredict[trainIndex] == yVec[trainIndex]):
            numTrainMatch += 1
    print("\n")
    print("Training Set Accuracy: %.6f" % (100*numTrainMatch/numTrainEx))
    input("Program paused. Press enter to continue.")

    # Display example images along with predictions from neural network
    randIndices = np.random.permutation(numTrainEx)
    for exampleIndex in range(0,10):
        print("\n")
        print("Displaying Example Image")
        print("\n")
        xMatSel = np.reshape(xMat[randIndices[exampleIndex],:],(1,numFeatures),order='F')
        returnCode = displayData(xMatSel)
        plt.show()
        examplePredict = predict(theta1Mat,theta2Mat,xMatSel)
        print("Neural Network Prediction: %d (digit %d)" % (examplePredict,examplePredict%10))
        input("Program paused. Press enter to continue.")

# Call main function
if __name__ == "__main__":
    main()
