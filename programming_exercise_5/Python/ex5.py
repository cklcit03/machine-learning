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
# Programming Exercise 5: Regularized Linear Regression and Bias vs. Variance
# Problem: Predict amount of water flowing out of a dam given data for 
# change of water level in a reservoir
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import fmin_bfgs

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

class InsufficientDegree(Exception):
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

    # Note that standard deviation for numpy uses a denominator of n
    # Standard deviation for R and Octave uses a denominator of n-1
    sigmaVec = np.std(X,axis=0,dtype=np.float32)
    for index in range(0,numTrainEx):
        xNormalized[index] = np.divide(np.subtract(X[index,:],muVec),sigmaVec)
    returnList = {'xNormalized': xNormalized,'muVec': muVec,'sigmaVec': sigmaVec}

    return(returnList)

# Compute regularized cost function J(\theta)
def computeCost(theta,X,y,lamb):
    "Compute regularized cost function J(\theta)"
    numFeatures = X.shape[1]
    if (numFeatures == 0):
        raise InsufficientFeatures('numFeatures = 0')
    theta = np.reshape(theta,(numFeatures,1),order='F')
    numTrainEx = y.shape[0]
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    diffVec = np.subtract(np.dot(X,theta),y)
    diffVecSq = np.multiply(diffVec,diffVec)
    jTheta = (np.sum(diffVecSq,axis=0))/(2.0*numTrainEx)
    thetaSquared = np.power(theta,2)
    jThetaReg = jTheta+(lamb/(2.0*numTrainEx))*(np.sum(thetaSquared,axis=0)-thetaSquared[0])

    return(jThetaReg)

# Compute gradient of regularized cost function J(\theta)
def computeGradient(theta,X,y,lamb):
    "Compute gradient of regularized cost function J(\theta)"
    numFeatures = X.shape[1]
    if (numFeatures == 0):
        raise InsufficientFeatures('numFeatures = 0')
    theta = np.reshape(theta,(numFeatures,1),order='F')
    numTrainEx = y.shape[0]
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    hTheta = np.dot(X,theta)
    gradArray = np.zeros((numFeatures,1))
    gradArrayReg = np.zeros((numFeatures,1))
    for gradIndex in range(0,numFeatures):
        gradTerm = np.multiply(np.reshape(X[:,gradIndex],(numTrainEx,1)),np.subtract(hTheta,y))
        gradArray[gradIndex] = (1/numTrainEx)*np.sum(gradTerm,axis=0)
        gradArrayReg[gradIndex] = gradArray[gradIndex]+(lamb/numTrainEx)*theta[gradIndex]

    gradArrayReg[0] = gradArrayReg[0]-(lamb/numTrainEx)*theta[0]
    gradArrayRegFlat = np.ndarray.flatten(gradArrayReg)
    return(gradArrayRegFlat)

# Train linear regression
def trainLinearReg(X,y,lamb):
    "Train linear regression"
    numFeatures = X.shape[1]
    if (numFeatures == 0):
        raise InsufficientFeatures('numFeatures = 0')
    initTheta = np.ones((numFeatures,1))
    initThetaFlat = np.ndarray.flatten(initTheta)
    fminBfgsOut = fmin_bfgs(computeCost,initThetaFlat,fprime=computeGradient,args=(X,y,lamb),maxiter=100,full_output=1)
    thetaOpt = np.reshape(fminBfgsOut[0],(numFeatures,1),order='F')
    return(thetaOpt)

# Generate values for learning curve
def learningCurve(X,y,XVal,yVal,lamb):
    "Generate values for learning curve"
    numTrainEx = y.shape[0]
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    errorTrain = np.zeros((numTrainEx,1))
    errorVal = np.zeros((numTrainEx,1))
    for exIndex in range(0,numTrainEx):
        XSubMat = X[0:(exIndex+1),:]
        ySubVec = y[0:(exIndex+1),:]
        trainThetaVec = trainLinearReg(XSubMat,ySubVec,1)
        errorTrain[exIndex] = computeCost(trainThetaVec,XSubMat,ySubVec,0)
        errorVal[exIndex] = computeCost(trainThetaVec,XVal,yVal,0)
    returnList = {'errorTrain': errorTrain,'errorVal': errorVal}

    return(returnList)

# Generate values for validation curve
def validationCurve(X,y,XVal,yVal):
    "Generate values for validation curve"
    lambdaVec = np.c_[0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
    numLambda = lambdaVec.shape[1]
    lambdaVec = np.reshape(lambdaVec,(numLambda,1),order='F')
    errorTrain = np.zeros((numLambda,1))
    errorVal = np.zeros((numLambda,1))
    for lambdaIndex in range(0,numLambda):
        currLambda = lambdaVec[lambdaIndex]
        trainThetaVec = trainLinearReg(X,y,currLambda)
        errorTrain[lambdaIndex] = computeCost(trainThetaVec,X,y,0)
        errorVal[lambdaIndex] = computeCost(trainThetaVec,XVal,yVal,0)
    returnList = {'lambdaVec': lambdaVec,'errorTrain': errorTrain,'errorVal': errorVal}

    return(returnList)

# Perform feature mapping for polynomial regression
def polyFeatures(X,p):
    "Perform feature mapping for polynomial regression"
    if (p <= 0):
        raise InsufficientDegree('p <= 0')
    XPoly = np.zeros((X.shape[0],p))
    for degIndex in range(0,p):
        XPoly[:,degIndex:(degIndex+1)] = np.power(X,degIndex+1)
    return(XPoly)

# Plot polynomial regression fit
def plotFit(minX,maxX,mu,sigma,theta,p):
    "Plot polynomial regression fit"
    xSeq = np.arange(minX-15,maxX+25,0.05)
    xSeqVec = np.reshape(xSeq,(xSeq.size,1),order='F')
    xPoly = polyFeatures(xSeqVec,p)
    xPolyNorm = np.zeros((xPoly.shape[0],p))
    for index in range(0,xPoly.shape[0]):
        xPolyNorm[index:(index+1),:] = np.divide(np.subtract(xPoly[index,:],mu),sigma)
    onesVec = np.ones((xPoly.shape[0],1))
    xPolyNorm = np.c_[onesVec,xPolyNorm]
    plt.plot(xSeqVec,np.dot(xPolyNorm,theta),'b-')

    return None

# Main function
def main():
    "Main function"
    print("Loading and Visualizing Data ...")
    waterTrainData = np.genfromtxt("../waterTrainData.txt",delimiter=",")
    numTrainEx = waterTrainData.shape[0]
    numFeatures = waterTrainData.shape[1]-1
    waterValData = np.genfromtxt("../waterValData.txt",delimiter=",")
    numValEx = waterValData.shape[0]
    waterTestData = np.genfromtxt("../waterTestData.txt",delimiter=",")
    numTestEx = waterTestData.shape[0]

    # Plot training data
    onesTrainVec = np.ones((numTrainEx,1))
    xMat = waterTrainData[:,0:numFeatures]
    yVec = waterTrainData[:,numFeatures:(numFeatures+1)]
    plt.plot(xMat,yVec,'rx',markersize=18)
    plt.ylabel('Water flowing out of the dam (y)',fontsize=18)
    plt.xlabel('Change in water level (x)',fontsize=18)
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Compute cost for regularized linear regression
    xMat = np.c_[onesTrainVec,xMat]
    thetaVec = np.ones((2,1))
    initCost = computeCost(thetaVec,xMat,yVec,1)
    print("Cost at theta = [1 ; 1]: %.6f" % initCost)
    print("(this value should be about 303.993192)")
    input("Program paused. Press enter to continue.")
    print("")

    # Compute gradient for regularized linear regression
    initGradient = computeGradient(thetaVec,xMat,yVec,1)
    print("Gradient at theta = [1 ; 1]: %s" % np.array_str(np.transpose(np.round(initGradient,6))))
    print("(this value should be about [-15.303016 598.250744])")
    input("Program paused. Press enter to continue.")
    print("")

    # Train linear regression
    lamb = 0
    trainThetaVec = trainLinearReg(xMat,yVec,lamb)

    # Plot fit over data
    plt.plot(xMat[:,1],yVec,'rx',markersize=18)
    plt.ylabel('Water flowing out of the dam (y)',fontsize=18)
    plt.xlabel('Change in water level (x)',fontsize=18)
    plt.hold(True)
    plt.plot(xMat[:,1],np.dot(xMat,trainThetaVec),'b-',markersize=18)
    plt.hold(False)
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Generate values for learning curve
    onesValVec = np.ones((numValEx,1))
    xValMat = np.c_[onesValVec,waterValData[:,0:numFeatures]]
    yValVec = waterValData[:,numFeatures:(numFeatures+1)]
    learningCurveList = learningCurve(xMat,yVec,xValMat,yValVec,lamb)

    # Plot learning curve
    plt.plot(np.arange(1,numTrainEx+1),learningCurveList['errorTrain'])
    plt.title('Learning curve for linear regression')
    plt.ylabel('Error',fontsize=18)
    plt.xlabel('Number of training examples',fontsize=18)
    plt.axis([0, 13, 0, 150])
    plt.hold(True)
    plt.plot(np.arange(1,numTrainEx+1),learningCurveList['errorVal'],color='g')
    plt.hold(False)
    plt.legend(('Train','Cross Validation'),loc='upper right')
    plt.show()
    print("")
    tab_head = ["# Training Examples","Train Error","Cross Validation Error"]
    print("\t\t".join(tab_head))
    for exIndex in range(0,numTrainEx):
        tab_line = [' ',exIndex+1,learningCurveList['errorTrain'][exIndex,],learningCurveList['errorVal'][exIndex,]]
        print(*tab_line,sep='\t\t')
    input("Program paused. Press enter to continue.")
    print("")

    # Perform feature mapping for polynomial regression
    p = 8
    xPoly = polyFeatures(xMat[:,1:2],p)
    xPolyNorm = featureNormalize(xPoly)
    xPolyNorm['xNormalized'] = np.c_[onesTrainVec,xPolyNorm['xNormalized']]
    xTestMat = waterTestData[:,0:numFeatures]
    xTestPoly = polyFeatures(xTestMat[:,0:1],p)
    xTestPolyNorm = np.zeros((numTestEx,p))
    for index in range(0,numTestEx):
        xTestPolyNorm[index] = np.divide(np.subtract(xTestPoly[index,:],xPolyNorm['muVec']),xPolyNorm['sigmaVec'])
    onesTestVec = np.ones((numTestEx,1))
    xTestPolyNorm = np.c_[onesTestVec,xTestPolyNorm]
    xValPoly = polyFeatures(xValMat[:,1:2],p)
    xValPolyNorm = np.zeros((numValEx,p))
    for index in range(0,numValEx):
        xValPolyNorm[index] = np.divide(np.subtract(xValPoly[index,:],xPolyNorm['muVec']),xPolyNorm['sigmaVec'])
    xValPolyNorm = np.c_[onesValVec,xValPolyNorm]
    print("Normalized Training Example 1:")
    print("%s\n" % np.array_str(np.round(np.transpose(xPolyNorm['xNormalized'][0:1,:]),6)))
    input("Program paused. Press enter to continue.")
    print("")

    # Train polynomial regression
    lamb = 0
    trainThetaVec = trainLinearReg(xPolyNorm['xNormalized'],yVec,lamb)

    # Plot fit over data
    plt.plot(xMat[:,1],yVec,'rx',markersize=18)
    plt.title('Polynomial Regression Fit (lambda = %f)' % lamb)
    plt.ylabel('Water flowing out of the dam (y)',fontsize=18)
    plt.xlabel('Change in water level (x)',fontsize=18)
    plt.axis([-100, 100, -80, 80])
    plt.hold(True)
    returnCode = plotFit(np.amin(xMat[:,1]),np.amax(xMat[:,1]),xPolyNorm['muVec'],xPolyNorm['sigmaVec'],trainThetaVec,p)
    plt.hold(False)
    plt.show()

    # Generate values for learning curve for polynomial regression
    learningCurveList = learningCurve(xPolyNorm['xNormalized'],yVec,xValPolyNorm,yValVec,lamb)

    # Plot learning curve
    plt.plot(np.arange(1,numTrainEx+1),learningCurveList['errorTrain'])
    plt.title('Polynomial Regression Learning Curve (lambda = %.6f)' % lamb)
    plt.ylabel('Error',fontsize=18)
    plt.xlabel('Number of training examples',fontsize=18)
    plt.axis([0, 13, 0, 100])
    plt.hold(True)
    plt.plot(np.arange(1,numTrainEx+1),learningCurveList['errorVal'],color='g')
    plt.hold(False)
    plt.legend(('Train','Cross Validation'),loc='upper right')
    plt.show()
    print("")
    print("Polynomial Regression (lambda = %.6f)" % lamb)
    print("")
    tab_head = ["# Training Examples","Train Error","Cross Validation Error"]
    print("\t\t".join(tab_head))
    for exIndex in range(0,numTrainEx):
        tab_line = [' ',exIndex+1,learningCurveList['errorTrain'][exIndex,],learningCurveList['errorVal'][exIndex,]]
        print(*tab_line,sep='\t\t')
    input("Program paused. Press enter to continue.")
    print("")

    # Generate values for validation curve for polynomial regression
    validationCurveList = validationCurve(xPolyNorm['xNormalized'],yVec,xValPolyNorm,yValVec)

    # Plot validation curve
    plt.plot(validationCurveList['lambdaVec'],validationCurveList['errorTrain'])
    plt.ylabel('Error',fontsize=18)
    plt.xlabel('lambda',fontsize=18)
    plt.hold(True)
    plt.plot(validationCurveList['lambdaVec'],validationCurveList['errorVal'],color='g')
    plt.hold(False)
    plt.legend(('Train','Cross Validation'),loc='upper right')
    plt.show()
    print("")
    tab_head = ["lambda","Train Error","Cross Validation Error"]
    print("\t\t".join(tab_head))
    for lambdaIndex in range(0,validationCurveList['lambdaVec'].shape[0]):
        tab_line = [validationCurveList['lambdaVec'][lambdaIndex],validationCurveList['errorTrain'][lambdaIndex,],validationCurveList['errorVal'][lambdaIndex,]]
        print(*tab_line,sep='\t\t')
    input("Program paused. Press enter to continue.")
    print("")

# Call main function
if __name__ == "__main__":
    main()
