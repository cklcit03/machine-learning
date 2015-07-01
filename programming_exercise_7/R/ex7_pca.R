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

FeatureNormalize <- function(X) {
  # Performs feature normalization.
  #
  # Args:
  #   X: Matrix of features.
  #
  # Returns:
  #   returnList: List of three objects.
  #               xNormalized: Matrix of normalized features.
  #               muVec: Vector of mean values of features.
  #               sigmaVec: Vector of standard deviations of features.
  numTrainEx <- dim(X)[1]
  numFeatures <- dim(X)[2]
  if (numFeatures >= 1) {
    if (numTrainEx >= 1) {
      xNormalized <- mat.or.vec(numTrainEx, numFeatures)
      muVec <- colMeans(X)
      sigmaVec <- t(rep(0, numFeatures))
      for (index in 1:numFeatures) {
        sigmaVec[index] <- sd(X[, index])
      }
      for (index in 1:numTrainEx) {
        xNormalized[index, ] <- (X[index, ] - muVec) / sigmaVec
      }
    } else {
      stop('Insufficient training examples')
    }
  } else {
    stop('Insufficient features')
  }
  returnList <- list("xNormalized"=xNormalized, "muVec"=muVec,
                     "sigmaVec"=sigmaVec)
  return(returnList)
}

Pca <- function(X) {
  # Runs PCA on input data.
  #
  # Args:
  #   X: Matrix of features.
  #
  # Returns:
  #   returnList: List of two objects.
  #               leftSingVec: Matrix whose columns contain the left singular
  #                            vectors of X.
  #               singVal: Vector containing the singular values of X.
  numTrainEx <- dim(X)[1]
  if (numTrainEx > 0) {
    covMat <- (1 / numTrainEx) * Conj(t(X)) %*% X
    svdX <- svd(covMat)
  } else {
    stop('Insufficient training examples')
  }
  returnList <- list("leftSingVec"=svdX$u, "singVal"=svdX$d)
  return(returnList)
}

ProjectData <- function(X, singVec, numDim) {
  # Projects input data onto reduced-dimensional space.
  #
  # Args:
  #   X: Matrix of features.
  #   singVec: Matrix whose columns contain the left singular vectors of X.
  #   numDim: Number of dimensions for dimensionality reduction.
  #
  # Returns:
  #   mappedData: Matrix of projected data, where each example has been
  #               projected onto the top numDim components of singVec.
  if (numDim > 0) {
    reducedSingVec <- singVec[, 1:numDim]
    mappedData <- X %*% reducedSingVec
  } else {
    stop('Insufficient dimensions')
  }
  return(mappedData)
}

RecoverData <- function(X, singVec, numDim) {
  # Projects input data onto original space.
  #
  # Args:
  #   X: Matrix of projected features.
  #   singVec: Matrix whose columns contain the left singular vectors of
  #            original features.
  #   numDim: Number of dimensions for projected features.
  #
  # Returns:
  #   recoveredData: Matrix of recovered data, where each projected example has
  #                  been projected onto the original high-dimensional space.
  if (numDim > 0) {
    reducedSingVec <- singVec[, 1:numDim]
    recoveredData <- X %*% Conj(t(reducedSingVec))
  } else {
    stop('Insufficient dimensions')
  }
  return(recoveredData)
}

DisplayData <- function(X) {
  # Displays 2D data in a grid.
  #
  # Args:
  #   X: Matrix of 2D data that will be displayed using image().
  #
  # Returns:
  #   None.
  numRows <- dim(X)[1]
  numCols <- dim(X)[2]
  if (numRows > 0) {
    if (numCols > 0) {
      exampleWidth <- round(sqrt(numCols))
      exampleHeight <- numCols / exampleWidth
      displayRows <- floor(sqrt(numRows))
      displayCols <- ceiling(numRows / displayRows)
      kPad <- 1
      displayArray <- matrix(-1, kPad + displayRows * (exampleHeight + kPad),
                             kPad + displayCols * (exampleWidth + kPad))
      currEx <- 1
      for (rowIndex in 1:displayRows) {
        for (colIndex in 1:displayCols) {
          if (currEx > numRows) {
            break
          }
          maxVal <- max(abs(X[currEx, ]))
          rowIdx <- kPad + (rowIndex - 1) * (exampleHeight + kPad) + 
            1:exampleHeight
          colIdx <- kPad + (colIndex - 1) * (exampleWidth + kPad) + 
            1:exampleWidth
          xReshape <- matrix(X[currEx, ], nrow=exampleHeight, byrow=FALSE)
          displayArray[rowIdx, colIdx] <- (1 / maxVal) * 
            as.numeric(t(xReshape[nrow(xReshape):1, ]))
          currEx <- currEx + 1
        }
        if (currEx > numRows) {
          break
        }
      }
      image(displayArray, col=gray.colors(256), axes=FALSE, zlim=c(-1, 1))
    } else {
      stop('Insufficient number of columns')
    }
  } else {
    stop('Insufficient number of rows')
  }
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
  segments(startPoint[1], startPoint[2], endPoint[1], endPoint[2], lwd=2)
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
print(sprintf("Visualizing example dataset for PCA."))
exercise7Data1 <- read.csv("../ex7data1.txt", header=FALSE)
xMat <- cbind(exercise7Data1[, "V1"], exercise7Data1[, "V2"])
numTrainEx <- dim(xMat)[1]

# Visualize input data
par(pty="s")
plot(x=xMat[, 1], y=xMat[, 2], col="blue", cex=1.75, pch=1, xlim=c(0.5, 6.5),
     ylim=c(2, 8), xlab="", ylab="")
returnCode <- ReadKey()

# Run PCA on input data
print(sprintf("Running PCA on example dataset."))
featureNormalizeList <- FeatureNormalize(xMat)
pcaList <- Pca(featureNormalizeList$xNormalized)

# Draw eigenvectors centered at mean of data
returnCode <- DrawLine(featureNormalizeList$muVec, 
                       featureNormalizeList$muVec + 1.5 * pcaList$singVal[1] * 
                         Conj(t(pcaList$leftSingVec[, 1])))
returnCode <- DrawLine(featureNormalizeList$muVec, 
                       featureNormalizeList$muVec + 1.5 * pcaList$singVal[2] * 
                         Conj(t(pcaList$leftSingVec[, 2])))
print(sprintf("Top eigenvector: "))
print(sprintf("U(:,1) = %f %f", pcaList$leftSingVec[1, 1], 
              pcaList$leftSingVec[2, 1]))
print(sprintf("(you should expect to see -0.707107 -0.707107)"))
returnCode <- ReadKey()

# Project data onto reduced-dimensional space
print(sprintf("Dimension reduction on example dataset."))
par(pty="s")
plot(x=featureNormalizeList$xNormalized[, 1], 
     y=featureNormalizeList$xNormalized[, 2], col="blue", cex=1.75, pch=1, 
     xlim=c(-4, 3), ylim=c(-4, 3), xlab="", ylab="")
kNumDim <- 1
projxMat <- ProjectData(featureNormalizeList$xNormalized, pcaList$leftSingVec, 
                        kNumDim)
print(sprintf("Projection of the first example: %f", projxMat[1]))
print(sprintf("(this value should be about 1.481274)"))
recovxMat <- RecoverData(projxMat, pcaList$leftSingVec, kNumDim)
print(sprintf("Approximation of the first example: %f %f", recovxMat[1, 1], 
              recovxMat[1, 2]))
print(sprintf("(this value should be about  -1.047419 -1.047419)"))

# Draw lines connecting projected points to original points
points(x=recovxMat[, 1], y=recovxMat[, 2], col="red", cex=1.75, pch=1, xlab="",
       ylab="")
for (exIndex in 1:numTrainEx) {
  returnCode <- DrawLine(featureNormalizeList$xNormalized[exIndex, ], 
                         recovxMat[exIndex, ])
}
returnCode <- ReadKey()

# Load and visualize face data
print(sprintf("Loading face dataset."))
exercise7Faces <- read.csv("../ex7faces.txt", header=FALSE)
facesMat <- as.matrix(exercise7Faces)
returnCode <- DisplayData(facesMat[1:100, ])
returnCode <- ReadKey()

# Run PCA on face data
print(sprintf("Running PCA on face dataset."))
print(sprintf("(this mght take a minute or two ...)"))
normalizedFacesList <- FeatureNormalize(facesMat)
facesList <- Pca(normalizedFacesList$xNormalized)

# Visualize top 36 eigenvectors for face data
returnCode <- DisplayData(Conj(t(facesList$leftSingVec[, 1:36])))
returnCode <- ReadKey()

# Project face data onto reduced-dimensional space
print(sprintf("Dimension reduction for face dataset."))
kNumFacesDim <- 100
projFaces <- ProjectData(normalizedFacesList$xNormalized, 
                         facesList$leftSingVec, kNumFacesDim)
print(sprintf("The projected data Z has a size of:"))
print(sprintf("%d %d", dim(projFaces)[1], dim(projFaces)[2]))
returnCode <- ReadKey()

# Visualize original (normalized) and projected face data side-by-side
print(sprintf("Visualizing the projected (reduced dimension) faces."))
recovFaces <- RecoverData(projFaces, facesList$leftSingVec, kNumFacesDim)
par(mfrow=c(1, 2))
returnCode <- DisplayData(normalizedFacesList$xNormalized[1:100, ])
title(main="Original faces")
returnCode <- DisplayData(recovFaces[1:100, ])
title(main="Recovered faces")
returnCode <- ReadKey()

# Use PCA for visualization of high-dimensional data
bird_small <- readPNG("../bird_small.png")
bird_small_reshape <- matrix(bird_small, 
                             nrow=(dim(bird_small)[1] * dim(bird_small)[2]))
kNumCentroids <- 16
kMaxIter <- 10
initialCentroids <- KMeansInitCentroids(bird_small_reshape, kNumCentroids)
kMeansList <- RunKMeans(bird_small_reshape, initialCentroids, kMaxIter, FALSE)
sampleIdx <- floor(runif(1000) * dim(bird_small_reshape)[1]) + 1
palette <- hsv(cbind((1 / (kNumCentroids + 1)) * 
                       seq(0, kNumCentroids, length=kNumCentroids + 1)), 1, 1)
currColors <- palette[kMeansList$centroidIndices[sampleIdx]]
dev.off()
plot.new()
par(new=TRUE)
scatterplot3d(bird_small_reshape[sampleIdx, 1], 
              bird_small_reshape[sampleIdx, 2], 
              bird_small_reshape[sampleIdx, 3], cex.symbols=1.75, pch=1, 
              color=currColors, xlab="", ylab="", zlab="", 
              main="Pixel dataset plotted in 3D. Color shows centroid 
                    memberships")
returnCode <- ReadKey()

# Project high-dimensional data to 2D for visualization
normalizedBirdList <- FeatureNormalize(bird_small_reshape)
birdList <- Pca(normalizedBirdList$xNormalized)
projBird <- ProjectData(normalizedBirdList$xNormalized, 
                        birdList$leftSingVec, 2)
returnCode <- PlotDataPoints(projBird[sampleIdx,], 
                             kMeansList$centroidIndices[sampleIdx], 
                             kNumCentroids)
title(main="Pixel dataset plotted in 2D, using PCA for dimensionality reduction")
returnCode <- ReadKey()
