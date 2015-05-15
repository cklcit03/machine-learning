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
# Programming Exercise 8: Anomaly Detection
# Problem: Apply anomaly detection to detect anomalous behavior in servers
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import png

class InsufficientData(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class InsufficientFeatures(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# Estimate mean and variance of input (Gaussian) data
def estimateGaussian(X):
    "Estimate mean and variance of input (Gaussian) data"
    numFeatures = X.shape[1]
    if (numFeatures > 0):
        muVec = np.mean(X,axis=0)
        varVec = np.var(X,axis=0,ddof=1)
    else:
        raise InsufficientFeatures('numFeatures <= 0')
    muVec = np.reshape(muVec,(1,numFeatures),order='F')
    varVec = np.reshape(varVec,(1,numFeatures),order='F')

    returnList = {'muVec': muVec,'varVec': varVec}

    return(returnList)

# Compute multivariate Gaussian PDF for input data
def multivariateGaussian(X,muVec,varVec):
    "Compute multivariate Gaussian PDF for input data"
    numFeatures = X.shape[1]
    if (numFeatures > 0):
        numData = X.shape[0]
        if (numData > 0):
            varMat = np.diag(np.ravel(varVec))
            probVec = np.zeros((numData,1))
            for dataIndex in range(0,numData):
                probVec[dataIndex] = (np.exp(-0.5*np.dot(np.dot(X[dataIndex,]-muVec,np.linalg.inv(varMat)),np.transpose(X[dataIndex,]-muVec))))/((np.power(2*np.pi,0.5*numFeatures)*np.sqrt(np.linalg.det(varMat))))
        else:
            raise InsufficientData('numData <= 0')
    else:
        raise InsufficientFeatures('numFeatures <= 0')

    return(probVec)

# Plot dataset and estimated Gaussian distribution
def visualizeFit(X,muVec,varVec):
    "Plot dataset and estimated Gaussian distribution"
    plt.scatter(X[:,0],X[:,1],s=80,marker='x',color='b')
    plt.ylabel('Throughput (mb/s)',fontsize=18)
    plt.xlabel('Latency (ms)',fontsize=18)
    plt.hold(True)
    uVals = np.linspace(0,35,num=71)
    vVals = np.linspace(0,35,num=71)
    zVals = np.zeros((uVals.shape[0],vVals.shape[0]))
    for uIndex in range(0,uVals.shape[0]):
        for vIndex in range(0,vVals.shape[0]):
            zVals[uIndex,vIndex] = multivariateGaussian(np.c_[uVals[uIndex],vVals[vIndex]],muVec,varVec)
    zValsTrans = np.transpose(zVals)
    uValsX,vValsY = np.meshgrid(uVals,vVals)
    zValsReshape = zValsTrans.reshape(uValsX.shape)
    expSeq = np.linspace(-20,1,num=8)
    powExpSeq = np.power(10,expSeq)
    plt.contour(uVals,vVals,zValsReshape,powExpSeq)
    plt.hold(False)

    return None

# Find the best threshold for detecting anomalies
def selectThreshold(y,p):
    "Find the best threshold for detecting anomalies"
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    stepSize = 0.001*(np.max(p)-np.min(p))
    epsilonSeq = np.linspace(np.min(p),np.max(p),num=np.ceil((np.max(p)-np.min(p))/stepSize))
    for epsilonIndex in range(0,1000):
        currEpsilon = epsilonSeq[epsilonIndex,]
        predictions = (p < currEpsilon)
        numTruePositives = np.sum((predictions == 1) & (y == 1))
        if (numTruePositives > 0):
            numFalsePositives = np.sum((predictions == 1) & (y == 0))
            numFalseNegatives = np.sum((predictions == 0) & (y == 1))
            precisionVal = numTruePositives/(numTruePositives+numFalsePositives)
            recallVal = numTruePositives/(numTruePositives+numFalseNegatives)
            F1 = (2*precisionVal*recallVal)/(precisionVal+recallVal)
            if (F1 > bestF1):
                bestF1 = F1
                bestEpsilon = currEpsilon

    returnList = {'bestF1': bestF1,'bestEpsilon': bestEpsilon}

    return(returnList)

# Main function
def main():
    "Main function"
    print("Visualizing example dataset for outlier detection.")
    serverData1 = np.genfromtxt("../serverData1.txt",delimiter=",")
    numFeatures = serverData1.shape[1]
    xMat = serverData1[:,0:numFeatures]
    plt.scatter(xMat[:,0],xMat[:,1],s=80,marker='x',color='b')
    plt.ylabel('Throughput (mb/s)',fontsize=18)
    plt.xlabel('Latency (ms)',fontsize=18)
    plt.axis([0, 30, 0, 30])
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Estimate (Gaussian) statistics of this dataset
    print("Visualizing Gaussian fit.")
    estimateGaussianList = estimateGaussian(xMat)
    muVec = estimateGaussianList['muVec']
    varVec = estimateGaussianList['varVec']
    probVec = multivariateGaussian(xMat,muVec,varVec)
    returnCode = visualizeFit(xMat,muVec,varVec)
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Use a cross-validation set to find outliers
    serverValData1 = np.genfromtxt("../serverValData1.txt",delimiter=",")
    numValFeatures = serverValData1.shape[1]-1
    xValMat = serverValData1[:,0:numValFeatures]
    yValVec = serverValData1[:,numValFeatures:(numValFeatures+1)]
    probValVec = multivariateGaussian(xValMat,muVec,varVec)
    selectThresholdList = selectThreshold(yValVec,probValVec)
    bestEpsilon = selectThresholdList['bestEpsilon']
    bestF1 = selectThresholdList['bestF1']
    print("Best epsilon found using cross-validation: %e" % bestEpsilon)
    print("Best F1 on Cross Validation Set:  %f" % bestF1)
    print("   (you should see a value epsilon of about 8.99e-05)")
    outlierIndices = (np.array(np.where(probVec < bestEpsilon)))[0,:]
    returnCode = visualizeFit(xMat,muVec,varVec)
    plt.hold(True)
    outliers = plt.scatter(xMat[outlierIndices,0],xMat[outlierIndices,1],s=80,marker='o',facecolors='none',edgecolors='r')
    plt.hold(False)
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Detect anomalies in another dataset
    serverData2 = np.genfromtxt("../serverData2.txt",delimiter=",")
    numFeatures = serverData2.shape[1]
    xMat = serverData2[:,0:numFeatures]

    # Estimate (Gaussian) statistics of this dataset
    estimateGaussianList = estimateGaussian(xMat)
    muVec = estimateGaussianList['muVec']
    varVec = estimateGaussianList['varVec']
    probVec = multivariateGaussian(xMat,muVec,varVec)

    # Use a cross-validation set to find outliers in this dataset
    serverValData2 = np.genfromtxt("../serverValData2.txt",delimiter=",")
    numValFeatures = serverValData2.shape[1]-1
    xValMat = serverValData2[:,0:numValFeatures]
    yValVec = serverValData2[:,numValFeatures:(numValFeatures+1)]
    probValVec = multivariateGaussian(xValMat,muVec,varVec)
    selectThresholdList = selectThreshold(yValVec,probValVec)
    bestEpsilon = selectThresholdList['bestEpsilon']
    bestF1 = selectThresholdList['bestF1']
    outlierIndices = (np.array(np.where(probVec < bestEpsilon)))[0,:]
    print("Best epsilon found using cross-validation: %e" % bestEpsilon)
    print("Best F1 on Cross Validation Set:  %f" % bestF1)
    print("# Outliers found: %d" % outlierIndices.size)
    print("   (you should see a value epsilon of about 1.38e-18)")
    input("Program paused. Press enter to continue.")
    print("")

# Call main function
if __name__ == "__main__":
    main()
