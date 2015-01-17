# Copyright (C) 2015  Caleb Lo
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
# Programming Exercise 6: Support Vector Machines
# Problem: Use SVMs to learn decision boundaries for various example datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn import svm

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

    return None

# Plot linear decision boundary
def plotLinearDecisionBoundary(x,y,theta):
    "Plot linear decision boundary"
    plotData(np.c_[x[:,0],x[:,1]],y)
    plt.hold(True)
    yLineVals = (theta[0]+theta[1]*x[:,0])/(-1*theta[2])
    plt.plot(x[:,0],yLineVals,'b-',markersize=18)
    plt.hold(False)

    return None

# Plot non-linear decision boundary learned via SVM
def plotDecisionBoundary(x,y,svmModel):
    "Plot non-linear decision boundary learned via SVM"
    plotData(np.c_[x[:,0],x[:,1]],y)
    plt.hold(True)
    x1 = np.linspace(np.amin(x[:,0],axis=0),np.amax(x[:,0],axis=0),num=100)
    x2 = np.linspace(np.amin(x[:,1],axis=0),np.amax(x[:,1],axis=0),num=100)
    jVals = np.zeros((x1.shape[0],x2.shape[0]))
    for x1Index in range(0,x1.shape[0]):
        for x2Index in range(0,x2.shape[0]):
            jVals[x1Index,x2Index] = svmModel.predict([[x1[x1Index,],x2[x2Index,]]])
    jValsTrans = np.transpose(jVals)
    x1X,x2Y = np.meshgrid(x1,x2)
    jValsReshape = jValsTrans.reshape(x1X.shape)
    plt.contour(x1,x2,jValsReshape,1)
    plt.hold(False)

    return None

# Select optimal learning parameters for radial basis SVM
def dataset3Params(X1,y1,Xval,yVal):
    "Select optimal learning parameters for radial basis SVM"
    C = 1
    sigma = 0.3
    cArr = np.c_[0.01,0.03,0.1,0.3,1,3,10,30]
    numC = cArr.shape[1]
    cArr= np.reshape(cArr,(numC,1),order='F')
    sigmaArr = np.c_[0.01,0.03,0.1,0.3,1,3,10,30]
    numSigma = sigmaArr.shape[1]
    sigmaArr= np.reshape(sigmaArr,(numSigma,1),order='F')
    bestPredErr = 1000000
    for cIndex in range(0,numC):
        for sigmaIndex in range(0,numSigma):
            svmModelTmp = svm.SVC(C=cArr[cIndex],kernel='rbf',gamma=1/(2*np.power(sigmaArr[sigmaIndex],2)))
            svmModelTmp.fit(X1,y1)
            predVec = svmModelTmp.predict(Xval)
            currPredErr = 0
            for valIndex in range(0,yVal.shape[0]):
                if (predVec[valIndex] != yVal[valIndex]):
                    currPredErr = currPredErr + 1
            if (currPredErr < bestPredErr):
                cBest = cArr[cIndex]
                sigmaBest = sigmaArr[sigmaIndex]
                bestPredErr = currPredErr
    C = cBest
    sigma = sigmaBest
    returnList = {'C': C,'sigma': sigma}

    return(returnList)

# Main function
def main():
    "Main function"
    print("Loading and Visualizing Data ...")
    exampleData1 = np.genfromtxt("../exampleData1.txt",delimiter=",")
    numTrainEx = exampleData1.shape[0]
    numFeatures = exampleData1.shape[1]-1

    # Plot data
    xMat = exampleData1[:,0:numFeatures]
    yVec = exampleData1[:,numFeatures]
    returnCode = plotData(xMat,yVec)
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Train linear SVM on data
    print("Training Linear SVM ...")
    svmModel = svm.SVC(kernel='linear')
    svmModel.fit(xMat,yVec)
    thetaOpt = np.c_[svmModel.intercept_,svmModel.coef_]
    returnCode = plotLinearDecisionBoundary(xMat,yVec,np.transpose(thetaOpt))
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Load another dataset and plot it
    exampleData2 = np.genfromtxt("../exampleData2.txt",delimiter=",")
    numTrainEx2 = exampleData2.shape[0]
    numFeatures2 = exampleData2.shape[1]-1
    xMat2 = exampleData2[:,0:numFeatures2]
    yVec2 = exampleData2[:,numFeatures2]
    returnCode = plotData(xMat2,yVec2)
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Train radial basis SVM on data
    print("Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...")
    sigmaVal = 0.1
    svmModel2 = svm.SVC(kernel='rbf',gamma=1/(2*np.power(sigmaVal,2)))
    svmModel2.fit(xMat2,yVec2)
    returnCode = plotDecisionBoundary(xMat2,yVec2,svmModel2)
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Load another dataset (along with cross-validation data) and plot it
    exampleData3 = np.genfromtxt("../exampleData3.txt",delimiter=",")
    numTrainEx3 = exampleData3.shape[0]
    numFeatures3 = exampleData3.shape[1]-1
    xMat3 = exampleData3[:,0:numFeatures3]
    yVec3 = exampleData3[:,numFeatures3]
    exampleValData3 = np.genfromtxt("../exampleValData3.txt",delimiter=",")
    numValEx3 = exampleValData3.shape[0]
    xValMat3 = exampleValData3[:,0:numFeatures3]
    yValVec3 = exampleValData3[:,numFeatures3]
    returnCode = plotData(xMat3,yVec3)
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Use cross-validation data to train radial basis SVM
    dataset3ParamsList = dataset3Params(xMat3,yVec3,xValMat3,yValVec3)
    svmModel3 = svm.SVC(C=dataset3ParamsList['C'],kernel='rbf',gamma=1/(2*np.power(dataset3ParamsList['sigma'],2)))
    svmModel3.fit(xMat3,yVec3)
    returnCode = plotDecisionBoundary(xMat3,yVec3,svmModel3)
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

# Call main function
if __name__ == "__main__":
    main()
