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
# Programming Exercise 7: Principal Component Analysis (PCA)
# Problem: Use PCA for dimensionality reduction
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import png
from mpl_toolkits.mplot3d import Axes3D

class InsufficientCentroids(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class InsufficientData(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class InsufficientDimensions(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

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

    # Note that standard deviation for numpy uses a denominator of n
    # Standard deviation for R and Octave uses a denominator of n-1
    sigmaVec = np.std(X,axis=0,dtype=np.float32)
    for index in range(0,numTrainEx):
        xNormalized[index] = np.divide(np.subtract(X[index,:],muVec),sigmaVec)
    returnList = {'xNormalized': xNormalized,'muVec': muVec,'sigmaVec': sigmaVec}

    return(returnList)

# Run PCA on input data
def pca(X):
    "Run PCA on input data"
    numTrainEx = X.shape[0]
    if (numTrainEx >= 1):
        covMat = (1/numTrainEx)*np.dot(X.conj().T,X)
        U,s,V = np.linalg.svd(covMat)
    else:
        raise InsufficientTrainingExamples('numTrainEx < 1')
    returnList = {"leftSingVec": U,"singVal": s}

    return(returnList)

# Draw line between input points
def drawLine(startPoint,endPoint):
    "Draw line between input points"
    plt.plot([startPoint[0],endPoint[0]],[startPoint[1],endPoint[1]],'b')

    return None

# Project input data onto reduced-dimensional space
def projectData(X,singVec,numDim):
    "Project input data onto reduced-dimensional space"
    if (numDim >= 1):
        reducedSingVec = singVec[:,0:numDim]
        mappedData = np.dot(X,reducedSingVec)
    else:
        raise InsufficientDimensions('numDim < 1')

    return(mappedData)

# Project input data onto original space
def recoverData(X,singVec,numDim):
    "Project input data onto original space"
    if (numDim >= 1):
        reducedSingVec = singVec[:,0:numDim]
        recoveredData = np.dot(X,reducedSingVec.T.conj())
    else:
        raise InsufficientDimensions('numDim < 1')

    return(recoveredData)

# Display 2D data in a grid
def displayData(X,axes=None):
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

    # If axes is not None, then we are using a subplot
    if (axes == None):
        plt.imshow(displayArray,cmap=cm.Greys_r)
        plt.axis('off')
    else:
        axes.imshow(displayArray,cmap=cm.Greys_r)
        axes.axis('off')

    return None

# Initialize centroids for input data
def kMeansInitCentroids(X,numCentroids):
    "Initialize centroids for input data"
    randIndices = np.random.permutation(X.shape[0])
    initCentroids = np.zeros((numCentroids,3))
    for centroidIdx in range(0,numCentroids):
        initCentroids[centroidIdx,:] = X[randIndices[centroidIdx,],:]
  
    return initCentroids

# Run K-Means Clustering on input data
def runKMeans(X,initCentroids,maxIter,plotFlag):
    "Run K-Means Clustering on input data"
    if (maxIter > 0):
        numData = X.shape[0]
        if (numData > 0):
            numCentroids = initCentroids.shape[0]
            if (numCentroids > 0):
                centroidIdx = np.zeros((numData,1))
                currCentroids = initCentroids

                # Create an array that stores all centroids
                # This array will be useful for plotting
                allCentroids = np.zeros((numCentroids*maxIter,initCentroids.shape[1]))
                for centroidIdx in range(0,numCentroids):
                    allCentroids[centroidIdx,:] = initCentroids[centroidIdx,:]
                for iterIndex in range(0,maxIter):
                    print("K-Means iteration %d/%d..." % (iterIndex+1,maxIter))

                    # Assign centroid to each datum
                    centroidIndices = findClosestCentroids(X,currCentroids)

                    # Plot progress of algorithm
                    if (plotFlag == True):
                        plotProgresskMeans(X,allCentroids,centroidIndices,numCentroids,iterIndex)
                        plt.show()
                        prevCentroids = currCentroids
                        input("Program paused. Press enter to continue.")
                        print("")

                    # Compute updated centroids
                    currCentroids = computeCentroids(X,centroidIndices,numCentroids)
                    if (iterIndex < (maxIter-1)):
                        for centroidIdx in range(0,numCentroids):
                            allCentroids[(iterIndex+1)*numCentroids+centroidIdx,:] = currCentroids[centroidIdx,:]
            else:
                raise InsufficientCentroids('numCentroids <= 0')
        else:
            raise InsufficientData('numData <= 0')
    else:
        raise InsufficientIterations('numiters <= 0')
    returnList = {'centroidIndices': centroidIndices,'currCentroids': currCentroids}

    return(returnList)

# Find closest centroids for input data using current centroid assignments
def findClosestCentroids(X,currCentroids):
    "Find closest centroids for input data using current centroid assignments"
    numCentroids = currCentroids.shape[0]
    if (numCentroids > 0):
        numData = X.shape[0]
        if (numData > 0):
            centroidIdx = np.zeros((numData,1))
            for dataIndex in range(0,numData):
                centroidIdx[dataIndex] = 0
                minDistance = np.sqrt(np.sum(np.multiply(X[dataIndex,:]-currCentroids[0,:],X[dataIndex,:]-currCentroids[0,:])))
                for centroidIndex in range(1,numCentroids):
                    tmpDistance = np.sqrt(np.sum(np.multiply(X[dataIndex,:]-currCentroids[centroidIndex,:],X[dataIndex,]-currCentroids[centroidIndex,:])))
                    if (tmpDistance < minDistance):
                        minDistance = tmpDistance
                        centroidIdx[dataIndex] = centroidIndex
        else:
            raise InsufficientData('numData <= 0')
    else:
        raise InsufficientCentroids('numCentroids <= 0')

    return(centroidIdx)

# Update centroids for input data using current centroid assignments
def computeCentroids(X,centroidIndices,numCentroids):
    "Update centroids for input data using current centroid assignments"
    if (numCentroids > 0):
        numFeatures = X.shape[1]
        if (numFeatures > 0):
            centroidArray = np.zeros((numCentroids,numFeatures))
            for centroidIndex in range(0,numCentroids):
                isCentroidIdx = (centroidIndices == centroidIndex)
                sumCentroidPoints = np.dot(np.transpose(isCentroidIdx),X)
                centroidArray[centroidIndex,:] = sumCentroidPoints/np.sum(isCentroidIdx)
        else:
            raise InsufficientFeatures('numFeatures <= 0')
    else:
        raise InsufficientCentroids('numCentroids <= 0')

    return(centroidArray)

# Display progress of K-Means Clustering
def plotProgresskMeans(X,allCentroids,centroidIndices,numCentroids,iterIndex):
    "Display progress of K-Means Clustering"
    if (numCentroids > 0):

        # Plot input data
        returnCode = plotDataPoints(X,centroidIndices,numCentroids)

        # Plot centroids as black X's
        centroids = plt.scatter(allCentroids[0:(iterIndex+1)*numCentroids,0],allCentroids[0:(iterIndex+1)*numCentroids,1],s=80,marker='x',color='k')

        # Plot history of centroids with lines
        for iter2Index in range(0,iterIndex):
            for centroidIndex in range(0,numCentroids):
                returnCode = drawLine(allCentroids[(iter2Index+1)*numCentroids+centroidIndex,:],allCentroids[iter2Index*numCentroids+centroidIndex,:])
    else:
        raise InsufficientCentroids('numCentroids <= 0')

    return None

# Plot input data with colors according to current cluster assignments
def plotDataPoints(X,centroidIndices,numCentroids):
    "Plot input data with colors according to current cluster assignments"
    palette = np.zeros((numCentroids+1,3))
    for centroidIdx in range(0,numCentroids+1):
        hsv_h = centroidIdx/(numCentroids+1)
        hsv_s = 1
        hsv_v = 1
        palette[centroidIdx,:] = colors.hsv_to_rgb(np.r_[hsv_h,hsv_s,hsv_v])
    numData = X.shape[0]
    currColors = np.zeros((numData,3))
    for dataIdx in range(0,numData):
        currCentroidIdx = centroidIndices[dataIdx].astype(int)
        currColors[currCentroidIdx,0] = palette[currCentroidIdx,0]
        currColors[currCentroidIdx,1] = palette[currCentroidIdx,1]
        currColors[currCentroidIdx,2] = palette[currCentroidIdx,2]
        plt.scatter(X[dataIdx,0],X[dataIdx,1],s=80,marker='o',facecolors='none',edgecolors=currColors[currCentroidIdx,:])

    return None

# Main function
def main():
    "Main function"
    print("Visualizing example dataset for PCA.")
    exercise7Data1 = np.genfromtxt("../ex7data1.txt",delimiter=",")
    numFeatures = exercise7Data1.shape[1]
    numTrainEx = exercise7Data1.shape[0]
    xMat = exercise7Data1[:,0:numFeatures]

    # Visualize input data
    plt.scatter(xMat[:,0],xMat[:,1],s=80,facecolors='none', edgecolors='b')
    axes = plt.gca()
    axes.set_xlim([0.5,6.5])
    axes.set_ylim([2,8])
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Run PCA on input data
    print("Running PCA on example dataset.")
    featureNormalizeList = featureNormalize(xMat)
    pcaList = pca(featureNormalizeList['xNormalized'])

    # Draw eigenvectors centered at mean of data
    plt.scatter(xMat[:,0],xMat[:,1],s=80,facecolors='none', edgecolors='b')
    axes = plt.gca()
    axes.set_xlim([0.5,6.5])
    axes.set_ylim([2,8])
    plt.hold(True)
    returnCode = drawLine(featureNormalizeList['muVec'],featureNormalizeList['muVec']+1.5*pcaList['singVal'][0]*pcaList['leftSingVec'][:,0])
    plt.hold(True)
    returnCode = drawLine(featureNormalizeList['muVec'],featureNormalizeList['muVec']+1.5*pcaList['singVal'][1]*pcaList['leftSingVec'][:,1])
    plt.show()
    print("Top eigenvector: ")
    print("U(:,1) = %f %f" % (pcaList['leftSingVec'][0,0],pcaList['leftSingVec'][1,0]))
    print("(you should expect to see -0.707107 -0.707107)")
    input("Program paused. Press enter to continue.")
    print("")

    # Project data onto reduced-dimensional space
    print("Dimension reduction on example dataset.")
    plt.scatter(featureNormalizeList['xNormalized'][:,0],featureNormalizeList['xNormalized'][:,1],s=80,facecolors='none', edgecolors='b')
    axes = plt.gca()
    axes.set_xlim([-4,3])
    axes.set_ylim([-4,3])
    numDim = 1
    projxMat = projectData(featureNormalizeList['xNormalized'],pcaList['leftSingVec'],numDim)
    print("Projection of the first example: %f" % projxMat[0])
    print("(this value should be about 1.481274)")
    recovxMat = recoverData(projxMat,pcaList['leftSingVec'],numDim)
    print("Approximation of the first example: %f %f" % (recovxMat[0,0],recovxMat[0,1]))
    print("(this value should be about  -1.047419 -1.047419)")

    # Draw lines connecting projected points to original points
    plt.hold(True)
    plt.scatter(recovxMat[:,0],recovxMat[:,1],s=80,facecolors='none', edgecolors='r')
    plt.hold(True)
    for exIndex in range(0,numTrainEx):
        returnCode = drawLine(featureNormalizeList['xNormalized'][exIndex,:],recovxMat[exIndex,:])
        plt.hold(True)
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Load and visualize face data
    print("Loading face dataset.")
    exercise7Faces = np.genfromtxt("../ex7faces.txt",delimiter=",")
    returnCode = displayData(exercise7Faces[0:100,:])
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Run PCA on face data
    print("Running PCA on face dataset.")
    print("(this mght take a minute or two ...)")
    normalizedFacesList = featureNormalize(exercise7Faces)
    facesList = pca(normalizedFacesList['xNormalized'])

    # Visualize top 36 eigenvectors for face data
    returnCode = displayData(facesList['leftSingVec'][:,0:36].T.conj())
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Project face data onto reduced-dimensional space
    print("Dimension reduction for face dataset.")
    numFacesDim = 100
    projFaces = projectData(normalizedFacesList['xNormalized'],facesList['leftSingVec'],numFacesDim)
    print("The projected data Z has a size of:")
    print("%d %d" % (projFaces.shape[0],projFaces.shape[1]))
    input("Program paused. Press enter to continue.")
    print("")

    # Visualize original (normalized) and projected face data side-by-side
    print("Visualizing the projected (reduced dimension) faces.")
    recovFaces = recoverData(projFaces,facesList['leftSingVec'],numFacesDim)
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.set_title('Original faces')
    returnCode = displayData(normalizedFacesList['xNormalized'][0:100,:],ax1)
    plt.hold(True)
    ax2.set_title('Recovered faces')
    returnCode = displayData(recovFaces[0:100,:],ax2)
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Use PCA for visualization of high-dimensional data
    birdSmallFile = open('../bird_small.png','rb')
    birdSmallReader = png.Reader(file=birdSmallFile)
    rowCount,colCount,birdSmall,meta = birdSmallReader.asDirect()
    planeCount = meta['planes']
    birdSmall2d = np.zeros((rowCount,colCount*planeCount))
    for rowIndex,oneBoxedRowFlatPixels in enumerate(birdSmall):
        birdSmall2d[rowIndex,:] = oneBoxedRowFlatPixels
        birdSmall2d[rowIndex,:] = (1/255)*birdSmall2d[rowIndex,:].astype(float)
    birdSmallReshape = birdSmall2d.reshape((rowCount*colCount,3))
    numCentroids = 16
    maxIter = 10
    initialCentroids = kMeansInitCentroids(birdSmallReshape,numCentroids)
    kMeansList = runKMeans(birdSmallReshape,initialCentroids,maxIter,False)
    sampleIdx = np.floor(np.random.uniform(size=1000)*birdSmallReshape.shape[0])
    palette = np.zeros((numCentroids+1,3))
    for centroidIdx in range(0,numCentroids+1):
        hsv_h = centroidIdx/(numCentroids+1)
        hsv_s = 1
        hsv_v = 1
        palette[centroidIdx,:] = colors.hsv_to_rgb(np.r_[hsv_h,hsv_s,hsv_v])
    fig = plt.figure(1)
    fig.clf()
    ax = Axes3D(fig)
    currColors = np.zeros((1000,3))
    for dataIdx in range(0,1000):
        currCentroidIdx = kMeansList['centroidIndices'][sampleIdx[dataIdx]].astype(int)
        currColors[currCentroidIdx,0] = palette[currCentroidIdx,0]
        currColors[currCentroidIdx,1] = palette[currCentroidIdx,1]
        currColors[currCentroidIdx,2] = palette[currCentroidIdx,2]
        ax.scatter(birdSmallReshape[sampleIdx[dataIdx].astype(int),0],birdSmallReshape[sampleIdx[dataIdx].astype(int),1],birdSmallReshape[sampleIdx[dataIdx].astype(int),2],s=80,marker='o',facecolors='none',edgecolors=currColors[currCentroidIdx,:])
    ax.set_title('Pixel dataset plotted in 3D. Color shows centroid memberships')
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

    # Project high-dimensional data to 2D for visualization
    normalizedBirdList = featureNormalize(birdSmallReshape)
    birdList = pca(normalizedBirdList['xNormalized'])
    projBird = projectData(normalizedBirdList['xNormalized'],birdList['leftSingVec'],2)
    returnCode = plotDataPoints(projBird[sampleIdx.astype(int),:],kMeansList['centroidIndices'][sampleIdx.astype(int),:],numCentroids)
    plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")

# Call main function
if __name__ == "__main__":
    main()
