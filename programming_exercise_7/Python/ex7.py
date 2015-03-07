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
# Programming Exercise 7: K-Means Clustering
# Problem: Apply K-Means Clustering to image compression
import itertools
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import png

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

# Draw line between input points
def drawLine(startPoint,endPoint):
    "Draw line between input points"
    plt.plot([startPoint[0],endPoint[0]],[startPoint[1],endPoint[1]],'b')

    return None

# Initialize centroids for input data
def kMeansInitCentroids(X,numCentroids):
    "Initialize centroids for input data"
    randIndices = np.random.permutation(X.shape[0])
    initCentroids = np.zeros((numCentroids,3))
    for centroidIdx in range(0,numCentroids):
        initCentroids[centroidIdx,:] = X[randIndices[centroidIdx,],:]
  
    return initCentroids

# Main function
def main():
    "Main function"
    print("Finding closest centroids.")
    exercise7Data2 = np.genfromtxt("../ex7data2.txt",delimiter=",")
    numFeatures = exercise7Data2.shape[1]
    xMat = exercise7Data2[:,0:numFeatures]

    # Select an initial set of centroids
    numCentroids = 3
    initialCentroids = np.r_[np.c_[3,3],np.c_[6,2],np.c_[8,5]]

    # Find closest centroids for example data using initial centroids
    centroidIndices = findClosestCentroids(xMat,initialCentroids)
    print("Closest centroids for the first 3 examples:")
    print("%s" % np.array_str(np.transpose(centroidIndices[0:3])+1))
    print("(the closest centroids should be 1, 3, 2 respectively)")
    input("Program paused. Press enter to continue.")
    print("")

    # Update centroids for example data
    print("Computing centroids means.")
    updatedCentroids = computeCentroids(xMat,centroidIndices,numCentroids)
    print("Centroids computed after initial finding of closest centroids:")
    print("%s" % np.array_str(np.round(updatedCentroids[0,:],6)))
    print("%s" % np.array_str(np.round(updatedCentroids[1,:],6)))
    print("%s" % np.array_str(np.round(updatedCentroids[2,:],6)))
    print("(the centroids should be")
    print("   [ 2.428301 3.157924 ]")
    print("   [ 5.813503 2.633656 ]")
    print("   [ 7.119387 3.616684 ]")
    input("Program paused. Press enter to continue.")
    print("")

    # Run K-Means Clustering on an example dataset
    print("Running K-Means clustering on example dataset.")
    maxIter = 10
    kMeansList = runKMeans(xMat,initialCentroids,maxIter,True)
    print("K-Means Done.")
    input("Program paused. Press enter to continue.")
    print("")

    # Use K-Means Clustering to compress an image
    print("Running K-Means clustering on pixels from an image.")
    birdSmallFile = open('../bird_small.png','rb')
    birdSmallReader = png.Reader(file=birdSmallFile)
    rowCount,colCount,birdSmall,meta = birdSmallReader.asDirect()
    planeCount = meta['planes']
    birdSmall2d = np.zeros((rowCount,colCount*planeCount))
    for rowIndex,oneBoxedRowFlatPixels in enumerate(birdSmall):
        birdSmall2d[rowIndex,:] = oneBoxedRowFlatPixels
        birdSmall2d[rowIndex,:] = (1/255)*birdSmall2d[rowIndex,:].astype(float)
    birdSmallReshape = birdSmall2d.reshape((rowCount*colCount,3))
    birdSmall3dReshape = birdSmallReshape.reshape((rowCount,colCount,3))
    numCentroids = 16
    maxIter = 10

    # Initialize centroids randomly
    initialCentroids = kMeansInitCentroids(birdSmallReshape,numCentroids)
    kMeansList = runKMeans(birdSmallReshape,initialCentroids,maxIter,False)
    input("Program paused. Press enter to continue.")
    print("")

    # Use the output clusters to compress this image
    print("Applying K-Means to compress an image.")
    currCentroids = kMeansList['currCentroids']
    centroidIndices = findClosestCentroids(birdSmallReshape,currCentroids)
    birdSmallRecovered = np.zeros((rowCount*colCount,3))
    for rowIndex in range(0,birdSmallRecovered.shape[0]):
        currIndex = centroidIndices[rowIndex,].astype(int)
        birdSmallRecovered[rowIndex,:] = currCentroids[currIndex,:]
    birdSmallRecovered = birdSmallRecovered.reshape((rowCount,colCount,3))

    # Display original and compressed images side-by-side
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.set_title('Original')
    im1 = ax1.imshow(birdSmall3dReshape)
    recoveredTitle = 'Compressed, with %d colors.' % numCentroids
    ax2.set_title(recoveredTitle)
    im2 = ax2.imshow(birdSmallRecovered)
    plt.show()

# Call main function
if __name__ == "__main__":
    main()
