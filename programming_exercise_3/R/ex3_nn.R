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
# Use parameters trained by a neural network for prediction

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

Predict <- function(Theta1, Theta2, X) {
  # Performs label prediction on training data.
  #
  # Args:
  #   Theta1: Matrix of neural network parameters (map from input layer to
  #           hidden layer).
  #   Theta2: Matrix of neural network parameters (map from hidden layer to
  #           output layer).
  #   X: Matrix of features.
  #
  # Returns:
  #   p: Vector of predicted class labels (one per example).
  numTrainEx <- dim(X)[1]
  onesVec <- t(t(rep(1, numTrainEx)))
  augX <- cbind(onesVec, X)
  hiddenLayerActivation <- ComputeSigmoid(augX %*% t(Theta1))
  onesVecMod <- t(t(rep(1, dim(hiddenLayerActivation)[1])))
  hiddenLayerActivationMod <- cbind(onesVecMod, hiddenLayerActivation)
  outputLayerActivation <- 
    ComputeSigmoid(hiddenLayerActivationMod %*% t(Theta2))
  p <- apply(outputLayerActivation, 1, which.max)
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

# Load two files that contain parameters trained by a neural network into R
print(sprintf("Loading Saved Neural Network Parameters ..."))
theta1 <- read.csv("../Theta1.txt", header=FALSE)
theta2 <- read.csv("../Theta2.txt", header=FALSE)
theta1Mat <- as.matrix(theta1)
theta2Mat <- as.matrix(theta2)

# Perform one-versus-all classification using trained parameters
trainingPredict <- (Predict(theta1Mat, theta2Mat, xMat))
print(sprintf("Training Set Accuracy: %.6f", 
              100 * apply((trainingPredict == yVec), 2, mean)))
returnCode <- ReadKey()

# Display example images along with predictions from neural network
randIndices <- randperm(numTrainEx, numTrainEx)
for (exampleIndex in 1:10) {
  print(sprintf("Displaying Example Image"))
  xMatSel <- subset(xMat, (rownames(xMat)) %in% randIndices[exampleIndex])
  returnCode <- DisplayData(xMatSel)
  examplePredict <- (Predict(theta1Mat, theta2Mat, xMatSel))
  print(sprintf("Neural Network Prediction: %d (digit %d)", examplePredict, 
                examplePredict %% 10))
  returnCode <- ReadKey()
}
