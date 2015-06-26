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
# Programming Exercise 3: (Multi-class) Logistic Regression and Neural Networks
# Problem: Predict label for a handwritten digit given data for 
# pixel values of various handwritten digits

# Load packages
library(pracma)

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

ComputeSigmoid <- function(z) {
  # Computes sigmoid function.
  #
  # Args:
  #   z: Can be a scalar, a vector or a matrix.
  #
  # Returns:
  #   sigmoidZ: Sigmoid function value.
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

OneVsAll <- function(X, y, numLabels, lambda) {
  # Trains multiple logistic regression classifiers.
  #
  # Args:
  #   X: Matrix of features.
  #   y: Vector of labels.
  #   numLabels: Number of classes.
  #   lambda: Regularization parameter.
  #
  # Returns:
  #   allTheta: Vector of regularized logistic regression parameters (one 
  #                 per class).
  if (numLabels > 0) {
    numTrainEx <- dim(X)[1]
    numFeatures <- dim(X)[2]
    allTheta <- matrix(0, numLabels, numFeatures + 1)
    onesVec <- t(t(rep(1, numTrainEx)))
    augX <- cbind(onesVec, X)
    for (labelIndex in 1:numLabels) {
      thetaVec <- t(t(rep(0, numFeatures + 1)))
      optimResult <- optim(thetaVec, fn=ComputeCost, gr=ComputeGradient, augX, 
                           as.numeric(yVec == labelIndex), numTrainEx, lambda,
                           method="BFGS", control=list(maxit=400))
      allTheta[labelIndex, ] <- optimResult$par
    }
  } else {
    stop('Insufficient number of classes')
  }
  return(allTheta)
}

PredictOneVsAll <- function(X, allTheta) {
  # Performs label prediction on training data.
  #
  # Args:
  #   X: Matrix of features.
  #   allTheta: Vector of regularized logistic regression parameters (one 
  #                 per class).
  #
  # Returns:
  #   p: Vector of predicted class labels (one per example).
  numTrainEx <- dim(X)[1]
  onesVec <- t(t(rep(1, numTrainEx)))
  augX <- cbind(onesVec, X)
  sigmoidArr <- ComputeSigmoid(augX %*% t(allTheta))
  p <- apply(sigmoidArr, 1, which.max)
  return(p)
}

# Use setwd() to set working directory to directory that contains this source 
# file
# Load file into R
print(sprintf("Loading and Visualizing Data ..."))
digitData <- read.csv("../digitData.txt", header=FALSE)
numTrainEx <- dim(digitData)[1]
xMat <- as.matrix(subset(digitData, select=-c(V401)))
yVec <- as.vector(subset(digitData, select=c(V401)))

# Randomly select 100 data points to display
randIndices <- randperm(numTrainEx, numTrainEx)
xMatSel <- subset(xMat, (rownames(xMat)) %in% randIndices[1])
for (randIndex in 2:100) {
  xMatSel <- rbind(xMatSel, subset(xMat, (rownames(xMat)) %in% 
                                   randIndices[randIndex]))
}
returnCode <- DisplayData(xMatSel)
returnCode <- ReadKey()

# Train one logistic regression classifier for each digit
print(sprintf("Training One-vs-All Logistic Regression..."))
kLambda <- 0.1
kNumLabels <- 10
allTheta <- OneVsAll(xMat, yVec, kNumLabels, kLambda)
returnCode <- ReadKey()

# Perform one-versus-all classification using logistic regression
trainingPredict <- (PredictOneVsAll(xMat, allTheta))
print(sprintf("Training Set Accuracy: %.6f", 
              100 * apply((trainingPredict == yVec), 2, mean)))
