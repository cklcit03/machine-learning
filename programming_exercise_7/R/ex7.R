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

ReadKey <- function() {
  # Reads key press.
  #
  # Args:
  #   None.
  #
  # Returns:
  #   None.
  cat("Program paused. Press enter to continue.")
  line <- readline()
  return(0)
}

FindClosestCentroids <- function(X, currCentroids) {
  # Finds closest centroids for input data using current centroid assignments.
  #
  # Args:
  #   X: Matrix of data features.
  #   currCentroids: Matrix of current centroid positions.
  #
  # Returns:
  #   centroidIdx: Vector where each entry contains index of closest centroid
  #                to corresponding example.
  numCentroids <- dim(currCentroids)[1]
  if (numCentroids > 0) {
    numData <- dim(X)[1]
    if (numData > 0) {
      centroidIdx <- t(t(rep(numData)))
      for (dataIndex in 1:numData) {
        centroidIdx[dataIndex] <- 1
        minDistance <- sqrt(sum((X[dataIndex, ] - currCentroids[1, ]) * 
                                (X[dataIndex, ] - currCentroids[1, ])))
        for (centroidIndex in 2:numCentroids) {
          tmpDistance <- 
            sqrt(sum((X[dataIndex, ] - currCentroids[centroidIndex, ]) * 
                     (X[dataIndex, ] - currCentroids[centroidIndex, ])))
          if (tmpDistance < minDistance) {
            minDistance <- tmpDistance
            centroidIdx[dataIndex] <- centroidIndex
          }
        }
      }
    } else {
      stop('Insufficient data')
    }
  } else {
    stop('Insufficient number of centroids')
  }
  return(centroidIdx)
}

ComputeCentroids <- function(X, centroidIndices, numCentroids) {
  # Updates centroids for input data using current centroid assignments.
  #
  # Args:
  #   X: Matrix of data features.
  #   centroidIndices: Vector where each entry contains index of closest 
  #                    centroid to corresponding example.
  #   numCentroids: Number of centroids.
  #
  # Returns:
  #   centroidArray: Matrix of centroid positions, where each centroid is the
  #                  mean of the points assigned to it.
  if (numCentroids > 0) {
    numFeatures <- dim(X)[2]
    if (numFeatures > 0) {
      centroidArray <- mat.or.vec(numCentroids, numFeatures)
      for (centroidIndex in 1:numCentroids) {
        isCentroidIdx <- (centroidIndices == centroidIndex)
        sumCentroidPoints <- isCentroidIdx %*% X
        centroidArray[centroidIndex, ] <- 
          sumCentroidPoints / sum(isCentroidIdx)
      }
    } else {
      stop('Insufficient number of features')
    }
  } else {
    stop('Insufficient number of centroids')
  }
  return(centroidArray)
}

RunKMeans <- function(X, initCentroids, maxIter, plotFlag) {
  # Runs K-Means Clustering on input data.
  #
  # Args:
  #   X: Matrix of data features.
  #   initCentroids: Matrix of initial centroid positions.
  #   maxIter: Maximum number of iterations for K-Means Clustering.
  #   plotFlag: Boolean that indicates whether progress of K-Means Clustering
  #             should be plotted.
  #
  # Returns:
  #   returnList: List of two objects.
  #               centroidIndices: Vector where each entry contains index of 
  #                                closest centroid to corresponding example.
  #               currCentroids: Matrix of centroid positions, where each 
  #                              centroid is the mean of the points assigned to 
  #                              it.
  if (maxIter > 0) {
    numData <- dim(X)[1]
    if (numData > 0) {
      numCentroids <- dim(initCentroids)[1]
      if (numCentroids > 0) {
        centroidIdx <- t(t(rep(numData)))
        currCentroids <- initCentroids
        prevCentroids <- currCentroids
        for (iterIndex in 1:maxIter) {
          print(sprintf("K-Means iteration %d/%d...", iterIndex, maxIter))

          # Assign centroid to each datum
          centroidIndices <- FindClosestCentroids(X, currCentroids)

          # Plot progress of algorithm
          if (plotFlag == TRUE) {
            PlotProgresskMeans(X, currCentroids, prevCentroids, 
                               centroidIndices, numCentroids, iterIndex)
            prevCentroids <- currCentroids
            returnCode <- ReadKey()
          }

          # Compute updated centroids
          currCentroids <- ComputeCentroids(X, centroidIndices, numCentroids)
        }
      } else {
        stop('Insufficient number of centroids')
      }
    } else {
      stop('Insufficient data')
    }
  } else {
    stop('Insufficient number of iterations')
  }
  returnList <- list("centroidIndices"=centroidIndices, 
                     "currCentroids"=currCentroids)
  return(returnList)
}

PlotProgresskMeans <- function(X, currCentroids, prevCentroids, 
                               centroidIndices, numCentroids, iterIndex) {
  # Displays progress of K-Means Clustering.
  #
  # Args:
  #   X: Matrix of data features.
  #   currCentroids: Matrix of current centroid positions.
  #   prevCentroids: Matrix of previous centroid positions.
  #   centroidIndices: Vector where each entry contains index of closest 
  #                    centroid to corresponding example.
  #   numCentroids: Number of centroids.
  #   iterIndex: Current iteration of K-Means Clustering.
  #
  # Returns:
  #   None.
  if (numCentroids > 0) {

    # Plot input data
    returnCode <- PlotDataPoints(X, centroidIndices, numCentroids)

    # Plot centroids as black X's
    points(x=currCentroids[, 1], y=currCentroids[, 2], col="black", cex=1.75, 
           pch=4, xlab="", ylab="")

    # Plot history of centroids with lines
    for (centroidIndex in 1:numCentroids) {
      returnCode <- DrawLine(currCentroids[centroidIndex, ], 
                             prevCentroids[centroidIndex, ])
    }
    par(new=TRUE)
  } else {
    stop('Insufficient number of centroids')
  }
  return(0)
}

PlotDataPoints <- function(X, centroidIndices, numCentroids) {
  # Plots input data with colors according to current cluster assignments.
  #
  # Args:
  #   X: Matrix of data features.
  #   centroidIndices: Vector where each entry contains index of closest 
  #                    centroid to corresponding example.
  #   numCentroids: Number of centroids.
  #
  # Returns:
  #   None.
  palette <- 
    hsv(cbind((1 / (numCentroids + 1)) * seq(0, numCentroids, 
                                             length=numCentroids + 1)), 1, 1)
  currColors <- palette[centroidIndices]
  plot(X[, 1], X[, 2], cex=1.75, pch=1, col=currColors, xlab="", ylab="", 
       main="")
  return(0)
}

DrawLine <- function(startPoint, endPoint) {
  # Draws line between input points.
  #
  # Args:
  #   startPoint: 2-D vector that represents starting point.
  #   endPoint: 2-D vector that represents ending point.
  #
  # Returns:
  #   None.
  segments(startPoint[1], startPoint[2], endPoint[1], endPoint[2])
  return(0)
}

KMeansInitCentroids <- function(X, numCentroids) {
  # Initializes centroids for input data.
  #
  # Args:
  #   X: Matrix of data features.
  #   numCentroids: Number of centroids.
  #
  # Returns:
  #   initCentroids: Matrix of initial centroid positions.
  randIndices <- sample(1:dim(X)[1])
  initCentroids <- X[t(t(randIndices))[1:numCentroids, ], ]
  return(initCentroids)
}

# Use setwd() to set working directory to directory that contains this source 
# file
# Load file into R
print(sprintf("Finding closest centroids."))
exercise7Data2 <- read.csv("../ex7data2.txt", header=FALSE)
xMat <- cbind(exercise7Data2[, "V1"], exercise7Data2[, "V2"])

# Select an initial set of centroids
numCentroids <- 3
initialCentroids <- rbind(cbind(3, 3), cbind(6, 2), cbind(8, 5))

# Find closest centroids for example data using initial centroids
centroidIndices <- FindClosestCentroids(xMat, initialCentroids)
print(sprintf("Closest centroids for the first 3 examples:"))
print(sprintf("%s", paste(format(round(centroidIndices[1:3], 0), nsmall=0), 
                          collapse=" ")))
print(sprintf("(the closest centroids should be 1, 3, 2 respectively)"))
returnCode <- ReadKey()

# Update centroids for example data
print(sprintf("Computing centroids means."))
updatedCentroids <- ComputeCentroids(xMat, centroidIndices, numCentroids)
print(sprintf("Centroids computed after initial finding of closest 
              centroids:"))
print(sprintf("%s", paste(format(round(updatedCentroids[1, ], 6), nsmall=6), 
                          collapse=" ")))
print(sprintf("%s", paste(format(round(updatedCentroids[2, ], 6), nsmall=6),
                          collapse=" ")))
print(sprintf("%s", paste(format(round(updatedCentroids[3, ], 6), nsmall=6),
                          collapse=" ")))
print(sprintf("(the centroids should be"))
print(sprintf("   [ 2.428301 3.157924 ]"))
print(sprintf("   [ 5.813503 2.633656 ]"))
print(sprintf("   [ 7.119387 3.616684 ]"))
returnCode <- ReadKey()

# Run K-Means Clustering on an example dataset
print(sprintf("Running K-Means clustering on example dataset."))
maxIter <- 10
kMeansList <- RunKMeans(xMat, initialCentroids, maxIter, TRUE)
print(sprintf("K-Means Done."))
returnCode <- ReadKey()

# Use K-Means Clustering to compress an image
print(sprintf("Running K-Means clustering on pixels from an image."))
bird_small <- readPNG("../bird_small.png")
bird_small_reshape <- matrix(bird_small, 
                             nrow=(dim(bird_small)[1] * dim(bird_small)[2]))
numCentroids <- 16
maxIter <- 10

# Initialize centroids randomly
initialCentroids <- KMeansInitCentroids(bird_small_reshape, numCentroids)
kMeansList <- RunKMeans(bird_small_reshape, initialCentroids, maxIter, FALSE)
returnCode <- ReadKey()

# Use the output clusters to compress this image
print(sprintf("Applying K-Means to compress an image."))
centroidIndices <- FindClosestCentroids(bird_small_reshape, 
                                        kMeansList$currCentroids)
bird_small_recovered <- kMeansList$currCentroids[centroidIndices, ]
bird_small_recovered <- array(bird_small_recovered, 
                              dim=c(dim(bird_small)[1], dim(bird_small)[2], 3))

# Display original and compressed images side-by-side
dev.off()
plot.new()
bird_small_raster <- as.raster(bird_small[, , 1:3])
rasterImage(bird_small_raster, 0, 0, 0.45, 0.9, interpolate=FALSE)
text(x=0.22, y=0.95, labels="Original")
bird_small_recovered_raster <- as.raster(bird_small_recovered[, , 1:3])
rasterImage(bird_small_recovered_raster, 0.55, 0,1, 0.9, interpolate=FALSE)
text(x=0.78, y=0.95, labels=sprintf("Compressed, with %d colors.", 
                                    numCentroids))
returnCode <- ReadKey()
