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

# Load packages
library(scatterplot3d)
library(png)

# Read key press
readKey <- function(){
  cat ("Program paused. Press enter to continue.")
  line <- readline()
  return(0)
}

# Perform feature normalization
featureNormalize <- function(X){
  numTrainEx = dim(X)[1]
  numFeatures = dim(X)[2]
  xNormalized = X
  muVec = colMeans(X)
  if (numFeatures >= 1) {
    sigmaVec = t(rep(0,numFeatures))
    for(index in 1:numFeatures) {
      sigmaVec[index] <- sd(X[,index])
    }
    if (numTrainEx >= 1) {
      for(index in 1:numTrainEx) {
        xNormalized[index,] = (X[index,]-muVec)/sigmaVec
      }
    }
    else
      stop('Insufficient training examples')
  }
  else
    stop('Insufficient features')
  returnList = list("xNormalized"=xNormalized,"muVec"=muVec,"sigmaVec"=sigmaVec)
  return(returnList)
}

# Run PCA on input data
pca <- function(X){
  numTrainEx = dim(X)[1]
  if (numTrainEx >= 1) {
    covMat = (1/numTrainEx)*Conj(t(X))%*%X
    svdX = svd(covMat)
  }
  else
    stop('Insufficient training examples')

  returnList = list("leftSingVec"=svdX$u,"singVal"=svdX$d)
  return(returnList)
}

# Draw line between input points
drawLine <- function(startPoint,endPoint) {
  segments(startPoint[1],startPoint[2],endPoint[1],endPoint[2],lwd=2)
  
  return(0)
}

# Project input data onto reduced-dimensional space
projectData <- function(X,singVec,numDim) {
  if (numDim >= 1) {
    reducedSingVec = singVec[,1:numDim]
    mappedData = X%*%reducedSingVec
  }
  else
    stop('Insufficient dimensions')
  
  return(mappedData)
}

# Project input data onto original space
recoverData <- function(X,singVec,numDim) {
  if (numDim >= 1) {
    reducedSingVec = singVec[,1:numDim]
    recoveredData = X%*%Conj(t(reducedSingVec))
  }
  else
    stop('Insufficient dimensions')
  
  return(recoveredData)
}

# Display 2D data in a grid
displayData <- function(X){
  exampleWidth = round(sqrt(dim(X)[2]))
  numRows = dim(X)[1]
  numCols = dim(X)[2]
  exampleHeight = (numCols/exampleWidth)
  displayRows = floor(sqrt(numRows))
  displayCols = ceiling(numRows/displayRows)
  pad = 1
  displayArray = matrix(-1,pad+displayRows*(exampleHeight+pad),pad+displayCols*(exampleWidth+pad))
  currEx = 1
  for(rowIndex in 1:displayRows) {
    for(colIndex in 1:displayCols) {
      if (currEx > numRows) {
        break
      }
      maxVal = max(abs(X[currEx,]))
      rowIdx = pad+(rowIndex-1)*(exampleHeight+pad)+1:exampleHeight
      colIdx = pad+(colIndex-1)*(exampleWidth+pad)+1:exampleWidth
      xReshape = matrix(X[currEx,],nrow=exampleHeight,byrow=FALSE)
      displayArray[rowIdx,colIdx] = (1/maxVal)*as.numeric(t(xReshape[nrow(xReshape):1,]))
      currEx = currEx+1
    }
    if (currEx > numRows) {
      break
    }
  }
  image(displayArray,col=gray.colors(256),axes=FALSE,zlim=c(-1,1))
  return(0)
}

# Initialize centroids for input data
kMeansInitCentroids <- function(X,numCentroids) {
  randIndices = sample(1:dim(X)[1])
  initCentroids = X[t(t(randIndices))[1:numCentroids,],]
  
  return(initCentroids)
}

# Run K-Means Clustering on input data
runKMeans <- function(X,initCentroids,maxIter,plotFlag){
  if (maxIter > 0) {
    numData = dim(X)[1]
    if (numData > 0) {
      numCentroids = dim(initCentroids)[1]
      if (numCentroids > 0) {
        centroidIdx = t(t(rep(numData)))
        currCentroids = initCentroids
        prevCentroids = currCentroids
        for(iterIndex in 1:maxIter) {
          print(sprintf("K-Means iteration %d/%d...",iterIndex,maxIter))
          
          # Assign centroid to each datum
          centroidIndices <- findClosestCentroids(X,currCentroids)
          
          # Plot progress of algorithm
          if (plotFlag == TRUE) {
            plotProgresskMeans(X,currCentroids,prevCentroids,centroidIndices,numCentroids,iterIndex)
            prevCentroids = currCentroids
            returnCode <- readKey()
          }
          
          # Compute updated centroids
          currCentroids <- computeCentroids(X,centroidIndices,numCentroids)
        }
      }
      else
        stop('Insufficient number of centroids')
    }
    else
      stop('Insufficient data')
  }
  else
    stop('Insufficient number of iterations')
  returnList = list("centroidIndices"=centroidIndices,"currCentroids"=currCentroids)
  return(returnList)
}

# Find closest centroids for input data using current centroid assignments
findClosestCentroids <- function(X,currCentroids){
  numCentroids = dim(currCentroids)[1]
  if (numCentroids > 0) {
    numData = dim(X)[1]
    if (numData >= 0) {
      centroidIdx = t(t(rep(numData)))
      for(dataIndex in 1:numData) {
        centroidIdx[dataIndex] = 1
        minDistance = sqrt(sum((X[dataIndex,]-currCentroids[1,])*(X[dataIndex,]-currCentroids[1,])))
        for(centroidIndex in 2:numCentroids) {
          tmpDistance = sqrt(sum((X[dataIndex,]-currCentroids[centroidIndex,])*(X[dataIndex,]-currCentroids[centroidIndex,])))
          if (tmpDistance < minDistance) {
            minDistance = tmpDistance
            centroidIdx[dataIndex] = centroidIndex
          }
        }
      }
    }
    else
      stop('Insufficient data')
  }
  else
    stop('Insufficient number of centroids')
  return(centroidIdx)
}

# Update centroids for input data using current centroid assignments
computeCentroids <- function(X,centroidIndices,numCentroids){
  if (numCentroids > 0) {
    numFeatures = dim(X)[2]
    if (numFeatures > 0) {
      centroidArray = mat.or.vec(numCentroids,numFeatures)
      for(centroidIndex in 1:numCentroids) {
        isCentroidIdx = (centroidIndices == centroidIndex)
        sumCentroidPoints = isCentroidIdx %*% X
        centroidArray[centroidIndex,] = sumCentroidPoints/sum(isCentroidIdx)
      }
    }
    else
      stop('Insufficient number of features')
  }
  else
    stop('Insufficient number of centroids')
  return(centroidArray)
}

# Display progress of K-Means Clustering
plotProgresskMeans <- function(X,currCentroids,prevCentroids,centroidIndices,numCentroids,iterIndex){
  if (numCentroids > 0) {
    
    # Plot input data
    returnCode <- plotDataPoints(X,centroidIndices,numCentroids)
    
    # Plot centroids as black X's
    points(x=currCentroids[,1],y=currCentroids[,2],col="black",cex=1.75,pch=4,xlab="",ylab="")
    
    # Plot history of centroids with lines
    for(centroidIndex in 1:numCentroids) {
      returnCode <- drawLine(currCentroids[centroidIndex,],prevCentroids[centroidIndex,])
    }
    par(new=TRUE)
  }
  else
    stop('Insufficient number of centroids')
  
  return(0)
}

# Plot input data with colors according to current cluster assignments
plotDataPoints <- function(X,centroidIndices,numCentroids) {
  palette = hsv(cbind((1/(numCentroids+1))*seq(0,numCentroids,length=numCentroids+1)),1,1)
  currColors = palette[centroidIndices]
  plot(X[,1],X[,2],cex=1.75,pch=1,col=currColors,xlab="",ylab="",main="")
  
  return(0)
}

# Use setwd() to set working directory to directory that contains this source file
# Load file into R
print(sprintf("Visualizing example dataset for PCA."))
exercise7Data1 = read.csv("../ex7data1.txt",header=FALSE)
xMat = cbind(exercise7Data1[,"V1"],exercise7Data1[,"V2"])
numTrainEx = dim(xMat)[1]

# Visualize input data
par(pty="s")
plot(x=xMat[,1],y=xMat[,2],col="blue",cex=1.75,pch=1,xlim=c(0.5,6.5),ylim=c(2,8),xlab="",ylab="")
returnCode <- readKey()

# Run PCA on input data
print(sprintf("Running PCA on example dataset."))
featureNormalizeList <- featureNormalize(xMat)
pcaList <- pca(featureNormalizeList$xNormalized)

# Draw eigenvectors centered at mean of data
returnCode <- drawLine(featureNormalizeList$muVec,featureNormalizeList$muVec+1.5*pcaList$singVal[1]*Conj(t(pcaList$leftSingVec[,1])))
returnCode <- drawLine(featureNormalizeList$muVec,featureNormalizeList$muVec+1.5*pcaList$singVal[2]*Conj(t(pcaList$leftSingVec[,2])))
print(sprintf("Top eigenvector: "))
print(sprintf("U(:,1) = %f %f",pcaList$leftSingVec[1,1],pcaList$leftSingVec[2,1]))
print(sprintf("(you should expect to see -0.707107 -0.707107)"))
returnCode <- readKey()

# Project data onto reduced-dimensional space
print(sprintf("Dimension reduction on example dataset."))
par(pty="s")
plot(x=featureNormalizeList$xNormalized[,1],y=featureNormalizeList$xNormalized[,2],col="blue",cex=1.75,pch=1,xlim=c(-4,3),ylim=c(-4,3),xlab="",ylab="")
numDim = 1
projxMat <- projectData(featureNormalizeList$xNormalized,pcaList$leftSingVec,numDim)
print(sprintf("Projection of the first example: %f",projxMat[1]))
print(sprintf("(this value should be about 1.481274)"))
recovxMat <- recoverData(projxMat,pcaList$leftSingVec,numDim)
print(sprintf("Approximation of the first example: %f %f",recovxMat[1,1],recovxMat[1,2]))
print(sprintf("(this value should be about  -1.047419 -1.047419)"))

# Draw lines connecting projected points to original points
points(x=recovxMat[,1],y=recovxMat[,2],col="red",cex=1.75,pch=1,xlab="",ylab="")
for(exIndex in 1:numTrainEx) {
  returnCode <- drawLine(featureNormalizeList$xNormalized[exIndex,],recovxMat[exIndex,])
}
returnCode <- readKey()

# Load and visualize face data
print(sprintf("Loading face dataset."))
exercise7Faces = read.csv("../ex7faces.txt",header=FALSE)
facesMat = as.matrix(exercise7Faces)
returnCode <- displayData(facesMat[1:100,])
returnCode <- readKey()

# Run PCA on face data
print(sprintf("Running PCA on face dataset."))
print(sprintf("(this mght take a minute or two ...)"))
normalizedFacesList <- featureNormalize(facesMat)
facesList <- pca(normalizedFacesList$xNormalized)

# Visualize top 36 eigenvectors for face data
returnCode <- displayData(Conj(t(facesList$leftSingVec[,1:36])))
returnCode <- readKey()

# Project face data onto reduced-dimensional space
print(sprintf("Dimension reduction for face dataset."))
numFacesDim = 100
projFaces <- projectData(normalizedFacesList$xNormalized,facesList$leftSingVec,numFacesDim)
print(sprintf("The projected data Z has a size of:"))
print(sprintf("%d %d",dim(projFaces)[1],dim(projFaces)[2]))
returnCode <- readKey()

# Visualize original (normalized) and projected face data side-by-side
print(sprintf("Visualizing the projected (reduced dimension) faces."))
recovFaces <- recoverData(projFaces,facesList$leftSingVec,numFacesDim)
par(mfrow=c(1,2))
returnCode <- displayData(normalizedFacesList$xNormalized[1:100,])
title(main="Original faces")
returnCode <- displayData(recovFaces[1:100,])
title(main="Recovered faces")
returnCode <- readKey()

# Use PCA for visualization of high-dimensional data
bird_small <- readPNG("../bird_small.png")
bird_small_reshape = matrix(bird_small,nrow=(dim(bird_small)[1]*dim(bird_small)[2]))
numCentroids = 16
maxIter = 10
initialCentroids <- kMeansInitCentroids(bird_small_reshape,numCentroids)
kMeansList <- runKMeans(bird_small_reshape,initialCentroids,maxIter,FALSE)
sampleIdx = floor(runif(1000)*dim(bird_small_reshape)[1])+1
palette = hsv(cbind((1/(numCentroids+1))*seq(0,numCentroids,length=numCentroids+1)),1,1)
currColors = palette[kMeansList$centroidIndices[sampleIdx]]
dev.off()
plot.new()
par(new=TRUE)
scatterplot3d(bird_small_reshape[sampleIdx,1],bird_small_reshape[sampleIdx,2],bird_small_reshape[sampleIdx,3],cex.symbols=1.75,pch=1,color=currColors,xlab="",ylab="",zlab="",main="Pixel dataset plotted in 3D. Color shows centroid memberships")
returnCode <- readKey()

# Project high-dimensional data to 2D for visualization
normalizedBirdList <- featureNormalize(bird_small_reshape)
birdList <- pca(normalizedBirdList$xNormalized)
projBird <- projectData(normalizedBirdList$xNormalized,birdList$leftSingVec,2)
returnCode <- plotDataPoints(projBird[sampleIdx,],kMeansList$centroidIndices[sampleIdx],numCentroids)
title(main="Pixel dataset plotted in 2D, using PCA for dimensionality reduction")
returnCode <- readKey()
