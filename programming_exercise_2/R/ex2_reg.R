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
# Programming Exercise 2: Logistic Regression
# Problem: Predict chances of acceptance for a microchip given data for 
# acceptance decisions and test scores of various microchips

PlotData <- function(X, y) {
  # Plots data.
  #
  # Args:
  #   X: Features to be plotted.
  #   y: Data labels.
  #
  # Returns:
  #   None.
  positiveIndices <- which(y == 1)
  negativeIndices <- which(y == 0)
  positiveExamples <- cbind(X[positiveIndices, ])
  negativeExamples <- cbind(X[negativeIndices, ])
  plot(positiveExamples[, 1], positiveExamples[, 2], cex=1.75, pch="+",
       xlab="", ylab="", xlim=c(-1, 1.5), ylim=c(-1, 1.5))
  points(x=negativeExamples[, 1], y=negativeExamples[, 2], col="yellow",
         bg="yellow", cex=1.75, pch=22, xlab="", ylab="")
  return(0)
}

PlotDecisionBoundary <- function(X, y, theta) {
  # Plots decision boundary.
  #
  # Args:
  #   X: Features that have already been plotted.
  #   y: Data labels.
  #   theta: Parameter that determines slope of decision boundary.
  #
  # Returns:
  #   None.
  returnCode <- PlotData(cbind(X[, 1], X[, 2]), y)
  u <- seq(-1, 1.5, length=50)
  v <- seq(-1, 1.5, length=50)
  z <- mat.or.vec(dim(cbind(u))[1], dim(cbind(v))[1])
  for (index0 in 1:dim(cbind(u))[1]) {
    for (index1 in 1:dim(cbind(v))[1]) {
      z[index0, index1] <- MapFeature(u[index0], v[index1]) %*% theta
    }
  }
  contour(x=u, y=v, z, nlevels=1, zlim=range(0), col="blue", lwd=2, 
          drawlabels=FALSE, add=TRUE)
  return(0)
}

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

MapFeature <- function(X1, X2) {
  # Adds polynomial features to training data.
  #
  # Args:
  #   X1: Vector of values for feature 1.
  #   X2: Vector of values for feature 2.
  #
  # Returns:
  #   augXMat: Vector of mapped features.
  numTrainEx <- dim(cbind(X1, X2))[1]
  if (numTrainEx > 0) {
    augXMat <- matrix(1, numTrainEx, 1)
    kDegree <- 6
    for (degIndex1 in 1:kDegree) {
      for (degIndex2 in 0:degIndex1) {
        augXMat <- cbind(augXMat, 
                         (X1 ^ (degIndex1 - degIndex2)) * (X2 ^ degIndex2))
      }
    }
  } else {
    stop('Insufficient training examples')
  }
  return(augXMat)
}

ComputeSigmoid <- function(z) {
  # Computes sigmoid function.
  #
  # Args:
  #   z: Can be a scalar, a vector or a matrix.
  #
  # Returns:
  #   None.
  sigmoidZ <- 1 / (1 + exp(-z))
  return(sigmoidZ)
}

ComputeCost <- function(theta, X, y, numTrainEx, lambda) {
  # Computes regularized cost function J(\theta).
  #
  # Args:
  #   theta: Vector of parameters for regularized logistic regression.
  #   X: Matrix of features.
  #   y: Vector of labels.
  #   numTrainEx: Number of training examples.
  #   lambda: Regularization parameter.
  #
  # Returns:
  #   jThetaReg: Regularized logistic regression cost.
  if (numTrainEx > 0) {
    hTheta <- ComputeSigmoid(X %*% theta)
    thetaSquared <- theta ^ 2
    jTheta <- (colSums(-y * log(hTheta) 
                       - (1 - y) * log(1 - hTheta))) / numTrainEx
    jThetaReg <- jTheta + (lambda / (2 * numTrainEx)) * sum(thetaSquared[-1])
  } else {
    stop('Insufficient training examples')
  }
  return(jThetaReg)
}

ComputeGradient <- function(theta, X, y, numTrainEx, lambda) {
  # Computes gradient of regularized cost function J(\theta).
  #
  # Args:
  #   theta: Vector of parameters for regularized logistic regression.
  #   X: Matrix of features.
  #   y: Vector of labels.
  #   numTrainEx: Number of training examples.
  #   lambda: Regularization parameter.
  #
  # Returns:
  #   gradArrayReg: Vector of regularized logistic regression gradients (one 
  #                 per feature).
  numFeatures <- dim(X)[2]
  if (numFeatures > 0) {
    if (numTrainEx > 0) {
      hTheta <- ComputeSigmoid(X %*% theta)
      gradArray <- matrix(0, numFeatures, 1)
      gradArrayReg <- matrix(0, numFeatures, 1)
      gradTermArray <- matrix(0, numTrainEx, numFeatures)
      for (gradIndex in 1:numFeatures) {
        gradTermArray[, gradIndex] <- (hTheta - y) * X[, gradIndex]
        gradArray[gradIndex] <- 
          (sum(gradTermArray[, gradIndex])) / (numTrainEx)
        gradArrayReg[gradIndex] <- gradArray[gradIndex] + 
          (lambda / numTrainEx) * theta[gradIndex]
      }
      gradArrayReg[1] <- gradArrayReg[1] - (lambda / numTrainEx) * theta[1]
    } else {
      stop('Insufficient training examples')
    }
  } else {
    stop('Insufficient features')
  }
  return(gradArrayReg)
}

ComputeCostGradList <- function(X, y, theta, lambda) {
  # Aggregates computed cost and gradient.
  #
  # Args:
  #   X: Matrix of features.
  #   y: Vector of labels.
  #   theta: Vector of parameters for regularized logistic regression.
  #   lambda: Regularization parameter.
  #
  # Returns:
  #   returnList: List of two objects.
  #               jThetaReg: Updated vector of parameters for regularized 
  #                          logistic regression.
  #               gradArrayReg: Updated vector of regularized logistic 
  #                             regression gradients (one per feature).
  numTrainEx <- dim(y)[1]
  jThetaReg <- ComputeCost(theta, X, y, numTrainEx, lambda)
  gradArrayReg <- ComputeGradient(theta, X, y, numTrainEx, lambda)
  returnList <- list("jThetaReg"=jThetaReg, "gradArrayReg"=gradArrayReg)
  return(returnList)
}

LabelPrediction <- function(X, theta){
  # Performs label prediction on training data.
  #
  # Args:
  #   X: Matrix of features.
  #   theta: Vector of parameters for regularized logistic regression.
  #
  # Returns:
  #   p: Vector of predictions (one per example).
  sigmoidArr <- ComputeSigmoid(X %*% theta)
  p <- (sigmoidArr >= 0.5)
  return(p)
}

# Use setwd() to set working directory to directory that contains this source 
# file
# Load file into R
microChipData <- read.csv("../microChipData.txt", header=FALSE)

# Plot data
returnCode <- PlotData(cbind(microChipData[, "V1"], microChipData[, "V2"]),
                       microChipData[, "V3"])
title(xlab="Microchip Test 1", ylab="Microchip Test 2")
plotLegend <- legend('bottomright', col=c("black", "yellow"),
                     pt.bg=c("black", "yellow"), pch=c(43, 22), pt.cex=1.75,
                     legend=c("", ""), bty="n", trace=TRUE)
text(plotLegend$text$x - 0.1, plotLegend$text$y, c('y = 1', 'y = 0'), pos=2)
numTrainEx <- dim(microChipData)[1]
numFeatures <- dim(microChipData)[2] - 1
xMat <- cbind(microChipData[, "V1"], microChipData[, "V2"])
yVec <- cbind(microChipData[, "V3"])

# Add polynomial features to training data
featureXMat <- MapFeature(xMat[, 1], xMat[, 2])
thetaVec <- t(t(rep(0, dim(featureXMat)[2])))

# Compute initial cost and gradient
kLambda <- 1
initComputeCostList <- ComputeCostGradList(featureXMat, yVec, thetaVec, 
                                           kLambda)
print(sprintf("Cost at initial theta (zeros): %.6f",
              initComputeCostList$jTheta))
returnCode <- ReadKey()

# Use optim to solve for optimum theta and cost
optimResult <- optim(thetaVec, fn=ComputeCost, gr=ComputeGradient, featureXMat,
                     yVec, numTrainEx, kLambda, method="BFGS",
                     control=list(maxit=400))

# Plot decision boundary
returnCode <- PlotDecisionBoundary(xMat, yVec, optimResult$par)
title(main=sprintf("lambda = %g", kLambda), xlab="Microchip Test 1",
      ylab="Microchip Test 2")
plotLegend <- legend('bottomright', col=c("black", "yellow"),
                     pt.bg=c("black", "yellow"), pch=c(43, 22), pt.cex=1.75,
                     legend=c("", ""), bty="n", trace=TRUE)
text(plotLegend$text$x - 0.1, plotLegend$text$y, c('y = 1', 'y = 0'), pos=2)

# Compute accuracy on training set
trainingPredict <- (LabelPrediction(featureXMat, optimResult$par) + 0)
print(sprintf("Train Accuracy: %.6f",
              100 * apply((trainingPredict == yVec), 2, mean)))
