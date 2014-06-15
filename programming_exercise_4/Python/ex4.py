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
# Programming Exercise 4: Multi-class Neural Networks
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

# Compute neural network cost
def computeCost(theta,X,y,lamb,layer1Size,layer2Size,layer3Size):
    "Compute neural network cost"
    theta = np.reshape(theta,(layer2Size*(layer1Size+1)+layer3Size*(layer2Size+1),1),order='F')
    theta1Slice = theta[0:(layer2Size*(layer1Size+1)),:]
    Theta1 = np.reshape(theta1Slice,(layer2Size,layer1Size+1),order='F')
    theta2Slice = theta[(layer2Size*(layer1Size+1)):(layer2Size*(layer1Size+1)+layer3Size*(layer2Size+1)),:]
    Theta2 = np.reshape(theta2Slice,(layer3Size,layer2Size+1),order='F')
    numTrainEx = X.shape[0]
    onesVec = np.ones((numTrainEx,1))
    augX = np.c_[onesVec,X]
    hiddenLayerActivation = computeSigmoid(np.dot(augX,np.transpose(Theta1)))
    onesVecMod = np.ones((hiddenLayerActivation.shape[0],1))
    hiddenLayerActivationMod = np.c_[onesVecMod,hiddenLayerActivation]
    outputLayerActivation = computeSigmoid(np.dot(hiddenLayerActivationMod,np.transpose(Theta2)))
    numLabels = Theta2.shape[0]
    yMat = np.zeros((numTrainEx,numLabels))
    for exampleIndex in range(0,numTrainEx):
        colVal = y[exampleIndex].astype(int)
        yMat[exampleIndex,colVal-1] = 1
    theta1Squared = np.power(Theta1,2)
    theta2Squared = np.power(Theta2,2)
    costTerm1 = np.multiply(-yMat,np.log(outputLayerActivation))
    costTerm2 = np.multiply(-(1-yMat),np.log(1-outputLayerActivation))
    jTheta = ((np.add(costTerm1,costTerm2)).sum())/numTrainEx
    jThetaReg = jTheta+(lamb/(2*numTrainEx))*((np.transpose(theta1Squared))[1:,].sum()+(np.transpose(theta2Squared))[1:,].sum())

    return(jThetaReg)

# Compute neural network gradient via backpropagation
def computeGradient(theta,X,y,lamb,layer1Size,layer2Size,layer3Size):
    "Compute neural network gradient via backpropagation"
    theta = np.reshape(theta,(layer2Size*(layer1Size+1)+layer3Size*(layer2Size+1),1),order='F')
    theta1Slice = theta[0:(layer2Size*(layer1Size+1)),:]
    Theta1 = np.reshape(theta1Slice,(layer2Size,layer1Size+1),order='F')
    theta2Slice = theta[(layer2Size*(layer1Size+1)):(layer2Size*(layer1Size+1)+layer3Size*(layer2Size+1)),:]
    Theta2 = np.reshape(theta2Slice,(layer3Size,layer2Size+1),order='F')
    numTrainEx = X.shape[0]
    onesVec = np.ones((numTrainEx,1))
    augX = np.c_[onesVec,X]
    delta1Mat = np.zeros((Theta1.shape[0],augX.shape[1]))
    delta2Mat = np.zeros((Theta2.shape[0],Theta1.shape[0]+1))
    numLabels = Theta2.shape[0]

    # Iterate over the training examples
    for exampleIndex in range(0,numTrainEx):

        # Step 1
        exampleX = augX[exampleIndex:(exampleIndex+1),:]
        hiddenLayerActivation = computeSigmoid(np.dot(exampleX,np.transpose(Theta1)))
        onesVecMod = np.ones((hiddenLayerActivation.shape[0],1))
        hiddenLayerActivationMod = np.c_[onesVecMod,hiddenLayerActivation]
        outputLayerActivation = computeSigmoid(np.dot(hiddenLayerActivationMod,np.transpose(Theta2)))

        # Step 2
        yVec = np.zeros((1,numLabels))
        colVal = y[exampleIndex].astype(int)
        yVec[:,colVal-1] = 1
        delta3Vec = np.transpose(np.subtract(outputLayerActivation,yVec))

        # Step 3
        delta2Int = np.dot(np.transpose(Theta2),delta3Vec)
        delta2Vec = np.multiply(delta2Int[1:,],computeSigmoidGradient(np.transpose(np.dot(exampleX,np.transpose(Theta1)))))

        # Step 4
        delta1Mat = np.add(delta1Mat,np.dot(delta2Vec,exampleX))
        delta2Mat = np.add(delta2Mat,np.dot(delta3Vec,np.c_[1,hiddenLayerActivation]))

    # Step 5 (without regularization)
    theta1Grad = (1/numTrainEx)*delta1Mat
    theta2Grad = (1/numTrainEx)*delta2Mat

    # Step 5 (with regularization)
    theta1Grad[:,1:] = theta1Grad[:,1:]+(lamb/numTrainEx)*Theta1[:,1:]
    theta2Grad[:,1:] = theta2Grad[:,1:]+(lamb/numTrainEx)*Theta2[:,1:]

    # Unroll gradients
    theta1GradStack = theta1Grad.flatten(1)
    theta2GradStack = theta2Grad.flatten(1)
    gradArrayReg = np.concatenate((theta1GradStack,theta2GradStack),axis=None)
    gradArrayRegFlat = np.ndarray.flatten(gradArrayReg)
    return(gradArrayRegFlat)

# Compute gradient of sigmoid function
def computeSigmoidGradient(z):
    "Compute gradient of sigmoid function"
    sigmoidGradientZ = np.multiply(computeSigmoid(z),1-computeSigmoid(z))

    return(sigmoidGradientZ)

# Initialize random weights of a neural network layer
def randInitializeWeights(lIn,lOut):
    "Initialize random weights of a neural network layer"
    epsilonInit = 0.12
    wMat = np.zeros((lOut,1+lIn))
    for lOutIndex in range(0,lOut):
        wMat[lOutIndex,:] = 2*epsilonInit*np.random.random((1+lIn,))-epsilonInit

    return(wMat)

# Perform label prediction on training data
def predict(Theta1,Theta2,X):
    "Perform label prediction on training data"
    numTrainEx = X.shape[0]
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
    inputLayerSize = theta1Mat.shape[1]-1
    hiddenLayerSize = theta2Mat.shape[1]-1
    numLabels = theta2Mat.shape[0]
    theta1MatStack = theta1Mat.flatten(1)
    theta2MatStack = theta2Mat.flatten(1)
    thetaStack = np.concatenate((theta1MatStack,theta2MatStack),axis=None)

    # Run feedforward section of neural network
    print("\n")
    print("Feedforward Using Neural Network ...")
    lamb = 0
    neuralNetworkCost = computeCost(thetaStack,xMat,yVec,lamb,inputLayerSize,hiddenLayerSize,numLabels)
    print("Cost at parameters (loaded from Theta1.txt and Theta2.txt): %.6f" % neuralNetworkCost)
    print("(this value should be about 0.287629)")
    print("\n")
    input("Program paused. Press enter to continue.")

    # Run feedforward section of neural network with regularization
    print("\n")
    print("Checking Cost Function (w/ Regularization) ...")
    lamb = 1
    neuralNetworkCost = computeCost(thetaStack,xMat,yVec,lamb,inputLayerSize,hiddenLayerSize,numLabels)
    print("Cost at parameters (loaded from Theta1.txt and Theta2.txt): %.6f" % neuralNetworkCost)
    print("(this value should be about 0.383770)")
    input("Program paused. Press enter to continue.")

    # Compute gradient for sigmoid function
    print("\n")
    print("Evaluating sigmoid gradient...")
    sigmoidGradient = computeSigmoidGradient(np.array([1,-0.5,0,0.5,1]))
    print("Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:")
    print("%s\n" % np.array_str(np.round(sigmoidGradient,6)))
    input("Program paused. Press enter to continue.")

    # Train neural network
    print("\n")
    print("Training Neural Network...")
    initTheta1 = randInitializeWeights(inputLayerSize,hiddenLayerSize)
    initTheta2 = randInitializeWeights(hiddenLayerSize,numLabels)
    initTheta1Stack = initTheta1.flatten(1)
    initTheta2Stack = initTheta2.flatten(1)
    thetaStack = np.concatenate((initTheta1Stack,initTheta2Stack),axis=None)
    thetaStackFlat = np.ndarray.flatten(thetaStack)
    fminNcgOut = fmin_ncg(computeCost,thetaStackFlat,fprime=computeGradient,args=(xMat,yVec,lamb,inputLayerSize,hiddenLayerSize,numLabels),avextol=1e-10,epsilon=1e-10,maxiter=20,full_output=1)
    thetaOpt = np.reshape(fminNcgOut[0],(hiddenLayerSize*(inputLayerSize+1)+numLabels*(hiddenLayerSize+1),1),order='F')
    theta1Slice = thetaOpt[0:(hiddenLayerSize*(inputLayerSize+1)),:]
    theta1Mat = np.reshape(theta1Slice,(hiddenLayerSize,inputLayerSize+1),order='F')
    theta2Slice = thetaOpt[(hiddenLayerSize*(inputLayerSize+1)):(hiddenLayerSize*(inputLayerSize+1)+numLabels*(hiddenLayerSize+1)),:]
    theta2Mat = np.reshape(theta2Slice,(numLabels,hiddenLayerSize+1),order='F')
    input("Program paused. Press enter to continue.")

    # Visualize neural network
    print("\n")
    print("Visualizing Neural Network...")
    returnCode = displayData(theta1Mat[:,1:])
    plt.show()
    print("\n")
    input("Program paused. Press enter to continue.")

    # Perform one-versus-all classification using trained parameters
    trainingPredict = predict(theta1Mat,theta2Mat,xMat)
    numTrainMatch = 0
    for trainIndex in range(0,numTrainEx):
        if (trainingPredict[trainIndex] == yVec[trainIndex]):
            numTrainMatch += 1
    print("\n")
    print("Training Set Accuracy: %.6f" % (100*numTrainMatch/numTrainEx))

# Call main function
if __name__ == "__main__":
    main()
