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

EstimateGaussian <- function(X) {
  # Estimates mean and variance of input (Gaussian) data.
  #
  # Args:
  #   X: Matrix of features.
  #
  # Returns:
  #   returnList: List of two objects.
  #               muVec: Vector of mean values of features.
  #               varVec: Vector of variances of features.
  numFeatures <- dim(X)[2]
  if (numFeatures > 0) {
    muVec <- colMeans(X)
    varVec <- t(rep(0, numFeatures))
    for (index in 1:numFeatures) {
      varVec[index] <- var(X[, index])
    } 
  } else {
    stop('Insufficient features')
  }
  returnList <- list("muVec"=as.matrix(muVec), "varVec"=t(varVec))
  return(returnList)
}

MultivariateGaussian <- function(X, muVec, varVec) {
  # Computes multivariate Gaussian PDF for input data.
  #
  # Args:
  #   X: Matrix of features.
  #   muVec: Vector of mean values of features.
  #   varVec: Vector of variances of features.
  #
  # Returns:
  #   probVec: Vector where each entry contains probability of corresponding 
  #            example.
  numFeatures <- dim(X)[2]
  if (numFeatures > 0) {
    numEx <- dim(X)[1]
    if (numEx > 0) {
      varMat <- diag(as.numeric(varVec))
      probVec <- t(t(rep(0, numEx)))
      for (exIndex in 1:numEx) {
        probVec[exIndex, ] <- (exp(-0.5 * t(X[exIndex, ] - muVec) %*% 
                                   solve(varMat) %*% (X[exIndex, ] - muVec))) / 
                              (((2 * pi) ^ (numFeatures / 2)) * 
                               sqrt(det(varMat)))
      }
    } else {
      stop('Insufficient examples')
    }
  } else {
    stop('Insufficient features')
  }
  return(probVec)
}

VisualizeFit <- function(X, muVec, varVec) {
  # Plots dataset and estimated Gaussian distribution.
  #
  # Args:
  #   X: Matrix of features.
  #   muVec: Vector of mean values of features.
  #   varVec: Vector of variances of features.
  #
  # Returns:
  #   probVec: Vector where each entry contains probability of corresponding 
  #            example.
  plot(X[, 1], X[, 2], col="blue", pch="x", xlab="Latency (ms)", 
       ylab="Throughput (mb/s)")
  u <- seq(0, 35, length=71)
  v <- seq(0, 35, length=71)
  z <- mat.or.vec(dim(cbind(u))[1], dim(cbind(v))[1])
  for (index0 in 1:dim(cbind(u))[1]) {
    for (index1 in 1:dim(cbind(v))[1]) {
      z[index0, index1] <- MultivariateGaussian(cbind(u[index0], v[index1]), 
                                                muVec, varVec)
    }
  }
  expSeq <- seq(-20, 1, length=8)
  powExpSeq <- 10 ^ expSeq
  contour(x=u, y=v, z, levels=powExpSeq, col="blue", lwd=1, drawlabels=FALSE, 
          add=TRUE)
  return(0)
}

SelectThreshold <- function(y, p){
  # Finds the best threshold for detecting anomalies.
  #
  # Args:
  #   y: Cross-validation labels.
  #   p: Vector where each entry contains probability of corresponding
  #      cross-validation example.
  #
  # Returns:
  #   returnList: List of two objects.
  #               bestF1: F1 score.
  #               bestEpsilon: Selected threshold.
  bestEpsilon <- 0
  bestF1 <- 0
  F1 <- 0
  stepSize <- 0.001 * (max(p) - min(p))
  epsilonSeq <- seq(min(p), max(p), by=stepSize)
  for (epsilonIndex in 1:length(epsilonSeq)) {
    currEpsilon <- epsilonSeq[epsilonIndex]
    predictions <- as.numeric((p < currEpsilon))
    numTruePositives <- sum((predictions == 1) & (y == 1))
    if (numTruePositives > 0) {
      numFalsePositives <- sum((predictions == 1) & (y == 0))
      numFalseNegatives <- sum((predictions == 0) & (y == 1))
      precisionVal <- numTruePositives / (numTruePositives + numFalsePositives)
      recallVal <- numTruePositives / (numTruePositives + numFalseNegatives)
      F1 <- (2 * precisionVal * recallVal) / (precisionVal + recallVal)
      if (F1 > bestF1) {
        bestF1 <- F1
        bestEpsilon <- currEpsilon
      }
    }
  }
  returnList <- list("bestF1"=bestF1, "bestEpsilon"=bestEpsilon)
  return(returnList)
}

# Use setwd() to set working directory to directory that contains this source file
# Load file into R
print(sprintf("Visualizing example dataset for outlier detection."))
serverData1 <- read.csv("../serverData1.txt", header=FALSE)
xMat <- cbind(serverData1[, "V1"], serverData1[, "V2"])
plot(serverData1[, "V1"], serverData1[, "V2"], col="blue", pch="x", 
     xlab="Latency (ms)", ylab="Throughput (mb/s)", xlim=c(0, 30), 
     ylim=c(0, 30))
returnCode <- ReadKey()

# Estimate (Gaussian) statistics of this dataset
print(sprintf("Visualizing Gaussian fit."))
estimateGaussianList <- EstimateGaussian(xMat)
muVec <- estimateGaussianList$muVec
varVec <- estimateGaussianList$varVec
probVec <- MultivariateGaussian(xMat, muVec, varVec)
returnCode <- VisualizeFit(xMat, muVec, varVec)
returnCode <- ReadKey()

# Use a cross-validation set to find outliers
serverValData1 <- read.csv("../serverValData1.txt", header=FALSE)
xValMat <- cbind(serverValData1[, "V1"], serverValData1[, "V2"])
yValVec <- cbind(serverValData1[, "V3"])
probValVec <- MultivariateGaussian(xValMat, muVec, varVec)
selectThresholdList <- SelectThreshold(yValVec, probValVec)
bestEpsilon <- selectThresholdList$bestEpsilon
bestF1 <- selectThresholdList$bestF1
print(sprintf("Best epsilon found using cross-validation: %e", bestEpsilon))
print(sprintf("Best F1 on Cross Validation Set:  %f", bestF1))
print(sprintf("   (you should see a value epsilon of about 8.99e-05)"))
outlierIndices <- which(probVec < bestEpsilon)
outlierExamples <- cbind(xMat[outlierIndices, ])
points(outlierExamples[, 1], outlierExamples[, 2], col="red", pch=1, cex=1.75)
returnCode <- ReadKey()

# Detect anomalies in another dataset
serverData2 <- read.csv("../serverData2.txt", header=FALSE)
xMat <- as.matrix(subset(serverData2, select=-c(12)))

# Estimate (Gaussian) statistics of this dataset
estimateGaussianList <- EstimateGaussian(xMat)
muVec <- estimateGaussianList$muVec
varVec <- estimateGaussianList$varVec
probVec <- MultivariateGaussian(xMat, muVec, varVec)

# Use a cross-validation set to find outliers in this dataset
serverValData2 <- read.csv("../serverValData2.txt", header=FALSE)
xValMat <- as.matrix(subset(serverValData2, select=-c(12)))
yValVec <- as.vector(subset(serverValData2, select=c(12)))
probValVec <- MultivariateGaussian(xValMat, muVec, varVec)
selectThresholdList <- SelectThreshold(yValVec, probValVec)
bestEpsilon <- selectThresholdList$bestEpsilon
bestF1 <- selectThresholdList$bestF1
print(sprintf("Best epsilon found using cross-validation: %e", bestEpsilon))
print(sprintf("Best F1 on Cross Validation Set:  %f", bestF1))
print(sprintf("# Outliers found: %d", sum(probVec < bestEpsilon)))
print(sprintf("   (you should see a value epsilon of about 1.38e-18)"))
returnCode <- ReadKey()
