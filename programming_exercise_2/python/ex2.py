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
# Problem: Predict chances of university admission for an applicant given data for 
# admissions decisions and test scores of various applicants
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
    plt.hold(False)
    plt.ylabel('Exam 2 score',fontsize=18)
    plt.xlabel('Exam 1 score',fontsize=18)

    return None

# Plot decision boundary
def plotDecisionBoundary(x,y,theta):
    "Plot decision boundary"
    plotData(np.c_[x[:,1],x[:,2]],y)
    plt.hold(True)
    yLineVals = (theta[0]+theta[1]*x[:,1])/(-1*theta[2])
    plt.plot(x[:,1],yLineVals,'b-',markersize=18)
    plt.hold(False)

    return None

# Compute sigmoid function
def computeSigmoid(z):
    "Compute sigmoid function"
    sigmoidZ = 1/(1+np.exp(-z))

    return(sigmoidZ)

# Compute cost function J(\theta)
def computeCost(theta,X,y,numTrainEx):
    "Compute cost function J(\theta)"
    numFeatures = X.shape[1]
    theta = np.reshape(theta,(numFeatures,1),order='F')
    hTheta = computeSigmoid(np.dot(X,theta))
    jTheta = (np.sum(np.subtract(np.multiply(-y,np.log(hTheta)),np.multiply((1-y),np.log(1-hTheta))),axis=0))/numTrainEx

    return(jTheta)

# Compute gradient of cost function J(\theta)
def computeGradient(theta,X,y,numTrainEx):
    "Compute gradient of cost function J(\theta)"
    numFeatures = X.shape[1]
    theta = np.reshape(theta,(numFeatures,1),order='F')
    hTheta = computeSigmoid(np.dot(X,theta))
    gradArray = np.zeros((numFeatures,1))
    if (numFeatures == 0):
        raise InsufficientFeatures('numFeatures = 0')
    for gradIndex in range(0,numFeatures):
        gradTerm = np.multiply(np.reshape(X[:,gradIndex],(numTrainEx,1)),np.subtract(hTheta,y))
        gradArray[gradIndex] = (1/numTrainEx)*np.sum(gradTerm,axis=0)

    gradArrayFlat = np.ndarray.flatten(gradArray)
    return(gradArrayFlat)

# Aggregate computed cost and gradient
def computeCostGradList(theta,X,y):
    "Aggregate computed cost and gradient"
    numFeatures = X.shape[1]
    numTrainEx = y.shape[0]
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    jTheta = computeCost(theta,X,y,numTrainEx)
    gradArrayFlat = computeGradient(theta,X,y,numTrainEx)
    gradArray = np.reshape(gradArrayFlat,(numFeatures,1),order='F')
    returnList = {'jTheta': jTheta,'gradArray': gradArray}

    return(returnList)

# Perform label prediction on training data
def labelPrediction(X,theta):
    "Perform label prediction on training data"
    numTrainEx = X.shape[0]
    sigmoidArray = computeSigmoid(np.dot(X,theta))
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
    print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")
    print("")
    applicantData = np.genfromtxt("../applicantData.txt",delimiter=",")
    returnCode = plotData(np.c_[applicantData[:,0],applicantData[:,1]],applicantData[:,2])
    plt.legend(('Admitted','Not admitted'),loc='lower right')
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")
    numTrainEx = applicantData.shape[0]
    numFeatures = applicantData.shape[1]-1
    onesVec = np.ones((numTrainEx,1))
    xMat = np.c_[applicantData[:,0],applicantData[:,1]]
    xMatAug = np.c_[onesVec,xMat]
    yVec = np.reshape(applicantData[:,2],(numTrainEx,1))
    thetaVec = np.zeros((numFeatures+1,1))

    # Compute initial cost and gradient
    initComputeCostList = computeCostGradList(thetaVec,xMatAug,yVec)
    print("Cost at initial theta (zeros): %.6f" % initComputeCostList['jTheta'])
    print("Gradient at initial theta (zeros):")
    print("%s\n" % np.array_str(np.round(initComputeCostList['gradArray'],6)))
    input("Program paused. Press enter to continue.")

    # Use fmin_ncg to solve for optimum theta and cost
    thetaVecFlat = np.ndarray.flatten(thetaVec)
    fminNcgOut = fmin_ncg(computeCost,thetaVecFlat,fprime=computeGradient,args=(xMatAug,yVec,numTrainEx),avextol=1e-10,epsilon=1e-10,maxiter=400,full_output=1)
    thetaOpt = np.reshape(fminNcgOut[0],(numFeatures+1,1),order='F')
    print("Cost at theta found by fmin_ncg: %.6f" % fminNcgOut[1])
    print("theta:")
    print("%s\n" % np.array_str(np.round(thetaOpt,6)))
    returnCode = plotDecisionBoundary(xMatAug,yVec,thetaOpt)
    plt.legend(('Decision Boundary','Admitted','Not admitted'),loc='lower left')
    plt.show()
    input("Program paused. Press enter to continue.")

    # Predict admission probability for a student with score 45 on exam 1 and score 85 on exam 2
    admissionProb = computeSigmoid(np.dot(np.array([1,45,85]),thetaOpt))
    print("For a student with scores 45 and 85, we predict an admission probability of %.6f" % admissionProb)

    # Compute accuracy on training set
    trainingPredict = labelPrediction(xMatAug,thetaOpt)
    numTrainMatch = 0
    for trainIndex in range(0,numTrainEx):
        if (trainingPredict[trainIndex] == yVec[trainIndex]):
            numTrainMatch += 1
    print("Train Accuracy: %.6f" % (100*numTrainMatch/numTrainEx))
    input("Program paused. Press enter to continue.")

# Call main function
if __name__ == "__main__":
    main()
