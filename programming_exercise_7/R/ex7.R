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

# Load packages
library(raster)
library(png)

# Read key press
readKey <- function(){
  cat ("Program paused. Press enter to continue.")
  line <- readline()
  return(0)
}

# Find closest centroids for input data using current centroid assignments
findClosestCentroids <- function(X,currCentroids){
  numCentroids = dim(currCentroids)[1]
  if (numCentroids > 0) {
    numData = dim(X)[1]
    if (numData > 0) {
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

# Draw line between input points
drawLine <- function(startPoint,endPoint) {
  segments(startPoint[1],startPoint[2],endPoint[1],endPoint[2])

  return(0)
}

# Initialize centroids for input data
kMeansInitCentroids <- function(X,numCentroids) {
  randIndices = sample(1:dim(X)[1])
  initCentroids = X[t(t(randIndices))[1:numCentroids,],]
  
  return(initCentroids)
}

# Use setwd() to set working directory to directory that contains this source file
# Load file into R
print(sprintf("Finding closest centroids."))
exercise7Data2 = read.csv("../ex7data2.txt",header=FALSE)
xMat = cbind(exercise7Data2[,"V1"],exercise7Data2[,"V2"])

# Select an initial set of centroids
numCentroids = 3
initialCentroids = rbind(cbind(3,3),cbind(6,2),cbind(8,5))

# Find closest centroids for example data using initial centroids
centroidIndices <- findClosestCentroids(xMat,initialCentroids)
print(sprintf("Closest centroids for the first 3 examples:"))
print(sprintf("%s",paste(format(round(centroidIndices[1:3],0),nsmall=0),collapse=" ")))
print(sprintf("(the closest centroids should be 1, 3, 2 respectively)"))
returnCode <- readKey()

# Update centroids for example data
print(sprintf("Computing centroids means."))
updatedCentroids <- computeCentroids(xMat,centroidIndices,numCentroids)
print(sprintf("Centroids computed after initial finding of closest centroids:"))
print(sprintf("%s",paste(format(round(updatedCentroids[1,],6),nsmall=6),collapse=" ")))
print(sprintf("%s",paste(format(round(updatedCentroids[2,],6),nsmall=6),collapse=" ")))
print(sprintf("%s",paste(format(round(updatedCentroids[3,],6),nsmall=6),collapse=" ")))
print(sprintf("(the centroids should be"))
print(sprintf("   [ 2.428301 3.157924 ]"))
print(sprintf("   [ 5.813503 2.633656 ]"))
print(sprintf("   [ 7.119387 3.616684 ]"))
returnCode <- readKey()

# Run K-Means Clustering on an example dataset
print(sprintf("Running K-Means clustering on example dataset."))
maxIter = 10
kMeansList <- runKMeans(xMat,initialCentroids,maxIter,TRUE)
print(sprintf("K-Means Done."))
returnCode <- readKey()

# Use K-Means Clustering to compress an image
print(sprintf("Running K-Means clustering on pixels from an image."))
bird_small <- readPNG("../bird_small.png")
bird_small_reshape = matrix(bird_small,nrow=(dim(bird_small)[1]*dim(bird_small)[2]))
numCentroids = 16
maxIter = 10

# Initialize centroids randomly
initialCentroids <- kMeansInitCentroids(bird_small_reshape,numCentroids)
kMeansList <- runKMeans(bird_small_reshape,initialCentroids,maxIter,FALSE)
returnCode <- readKey()

# Use the output clusters to compress this image
print(sprintf("Applying K-Means to compress an image."))
centroidIndices <- findClosestCentroids(bird_small_reshape,kMeansList$currCentroids)
bird_small_recovered = kMeansList$currCentroids[centroidIndices,]
bird_small_recovered = array(bird_small_recovered,dim=c(dim(bird_small)[1],dim(bird_small)[2],3))

# Display original and compressed images side-by-side
dev.off()
plot.new()
bird_small_raster <- as.raster(bird_small[,,1:3])
rasterImage(bird_small_raster,0,0,0.45,0.9,interpolate=FALSE)
text(x=0.22,y=0.95,labels="Original")
bird_small_recovered_raster <- as.raster(bird_small_recovered[,,1:3])
rasterImage(bird_small_recovered_raster,0.55,0,1,0.9,interpolate=FALSE)
text(x=0.78,y=0.95,labels=sprintf("Compressed, with %d colors.",numCentroids))
returnCode <- readKey()
