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
# Programming Exercise 4: Multi-class Neural Networks
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

ComputeCost <- function(theta, X, y, lambda, layer1Size, layer2Size, 
                        layer3Size) {
  # Computes neural network cost.
  #
  # Args:
  #   theta: Vector of neural network parameters.
  #   X: Matrix of features.
  #   y: Vector of labels.
  #   lambda: Regularization parameter.
  #   layer1Size: Number of units in input layer.
  #   layer2Size: Number of units in hidden layer.
  #   layer3Size: Number of units in output layer.
  #
  # Returns:
  #   jThetaReg: Regularized neural network cost.
  numTrainEx <- dim(X)[1]
  if (numTrainEx > 0) {
    thetaMat <- as.matrix(theta)
    Theta1 <- matrix(thetaMat[1:(layer2Size * (layer1Size + 1)), 1],
                     nrow=layer2Size)
    Theta2 <- matrix(thetaMat[(1 + layer2Size * (layer1Size + 1)):
                              (layer2Size * (layer1Size + 1) + 
                               layer3Size * (layer2Size + 1)), 1], 
                     nrow=layer3Size)
    onesVec <- t(t(rep(1, numTrainEx)))
    augX <- cbind(onesVec, X)
    hiddenLayerActivation <- ComputeSigmoid(augX %*% t(Theta1))
    onesVecMod <- t(t(rep(1, dim(hiddenLayerActivation)[1])))
    hiddenLayerActivationMod <- cbind(onesVecMod, hiddenLayerActivation)
    outputLayerActivation <- ComputeSigmoid(hiddenLayerActivationMod %*% 
                                            t(Theta2))
    numLabels <- dim(Theta2)[1]
    yMat <- mat.or.vec(numTrainEx, numLabels)
    for (exampleIndex in 1:numTrainEx) {
      yMat[exampleIndex, y[exampleIndex, ]] <- 1
    }
    costTerm1 <- -yMat * log(outputLayerActivation)
    costTerm2 <- -(1 - yMat) * log(1 - outputLayerActivation)
    jTheta <- (1 / numTrainEx) * sum(costTerm1 + costTerm2)
    theta1Squared <- Theta1 ^ 2
    theta2Squared <- Theta2 ^ 2
    jThetaReg <- jTheta + (lambda / (2 * numTrainEx)) * 
      (sum(t(theta1Squared)[-1, ]) + sum(t(theta2Squared)[-1, ]))
  } else {
    stop('Insufficient training examples')
  }
  return(jThetaReg)
}

ComputeGradient <- function(theta, X, y, lambda, layer1Size, layer2Size, 
                            layer3Size) {
  # Computes neural network gradient via backpropagation.
  #
  # Args:
  #   theta: Vector of neural network parameters.
  #   X: Matrix of features.
  #   y: Vector of labels.
  #   lambda: Regularization parameter.
  #   layer1Size: Number of units in input layer.
  #   layer2Size: Number of units in hidden layer.
  #   layer3Size: Number of units in output layer.
  #
  # Returns:
  #   gradArrayReg: Vector of regularized neural network gradients (one 
  #                 per feature).
  numTrainEx <- dim(X)[1]
  if (numTrainEx > 0) {
    thetaMat <- as.matrix(theta)
    Theta1 <- matrix(thetaMat[1:(layer2Size * (layer1Size + 1)), 1],
                     nrow=layer2Size)
    Theta2 <- matrix(thetaMat[(1 + layer2Size * (layer1Size + 1)):
                              (layer2Size * (layer1Size + 1) + 
                               layer3Size * (layer2Size + 1)), 1], 
                     nrow=layer3Size)
    onesVec <- t(t(rep(1, numTrainEx)))
    augX <- cbind(onesVec, X)
    delta1Mat <- mat.or.vec(dim(Theta1)[1], dim(augX)[2])
    delta2Mat <- mat.or.vec(dim(Theta2)[1], dim(Theta1)[1] + 1)
    numLabels <- dim(Theta2)[1]

    # Iterate over the training examples
    for (exampleIndex in 1:numTrainEx) {

      # Step 1
      exampleX <- augX[exampleIndex, ]
      hiddenLayerActivation <- ComputeSigmoid(exampleX %*% t(Theta1))
      onesVecMod <- t(t(rep(1, dim(hiddenLayerActivation)[1])))
      hiddenLayerActivationMod <- cbind(onesVecMod, hiddenLayerActivation)
      outputLayerActivation <- ComputeSigmoid(hiddenLayerActivationMod %*% 
                                              t(Theta2))

      # Step 2
      yVec <- mat.or.vec(1, numLabels)
      yVec[1, y[exampleIndex, ]] <- 1
      delta3Vec <- t(outputLayerActivation - yVec)

      # Step 3
      delta2Int <- t(Theta2) %*% delta3Vec
      delta2Vec <- delta2Int[-1, ] * ComputeSigmoidGradient(t(exampleX %*% 
                                                              t(Theta1)))

      # Step 4
      delta1Mat <- delta1Mat + delta2Vec %*% exampleX
      delta2Mat <- delta2Mat + delta3Vec %*% cbind(1, hiddenLayerActivation)
    }

    # Step 5 (without regularization)
    theta1Grad <- (1 / numTrainEx) * delta1Mat
    theta2Grad <- (1 / numTrainEx) * delta2Mat

    # Step 5 (with regularization)
    theta1Grad[, -1] <- theta1Grad[, -1] + (lambda / numTrainEx) * Theta1[, -1]
    theta2Grad[, -1] <- theta2Grad[, -1] + (lambda / numTrainEx) * Theta2[, -1]

    # Unroll gradients
    theta1GradStack <- stack(as.data.frame(theta1Grad))
    theta2GradStack <- stack(as.data.frame(theta2Grad))
    theta1GradStackVals <- theta1GradStack[, "values"]
    theta2GradStackVals <- theta2GradStack[, "values"]
    gradArrayReg <- c(t(theta1GradStackVals), t(theta2GradStackVals))
  } else {
    stop('Insufficient training examples')
  }
  return(gradArrayReg)
}

ComputeSigmoidGradient <- function(z) {
  # Computes gradient of sigmoid function.
  #
  # Args:
  #   z: Can be a scalar, a vector or a matrix.
  #
  # Returns:
  #   sigmoidGradientZ: Sigmoid gradient function value.
  sigmoidGradientZ <- ComputeSigmoid(z) * (1 - ComputeSigmoid(z))
  return(sigmoidGradientZ)
}

RandInitializeWeights <- function(lIn, lOut) {
  # Initializes random weights of neural network between layers $L$ and $L+1$.
  #
  # Args:
  #   lIn: Number of units in layer $L$.
  #   lOut: Number of units in layer $L+1$.
  #
  # Returns:
  #   wMat: Matrix of random weights.
  if (lOut > 0) {
    kEpsilonInit <- 0.12
    wMat <- mat.or.vec(lOut, 1 + lIn)
    for (lOutIndex in 1:lOut) {
      wMat[lOutIndex, ] <- 2 * kEpsilonInit * runif(1 + lIn) - kEpsilonInit
    }
  } else {
    stop('Insufficient number of units in layer $L+1$')
  }
  return(wMat)
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
  xMatSel <- rbind(xMatSel, 
                   subset(xMat, (rownames(xMat)) %in% randIndices[randIndex]))
}
returnCode <- DisplayData(xMatSel)
returnCode <- ReadKey()

# Load two files that contain parameters trained by a neural network into R
print(sprintf("Loading Saved Neural Network Parameters ..."))
theta1 <- read.csv("../Theta1.txt", header=FALSE)
theta2 <- read.csv("../Theta2.txt", header=FALSE)
theta1Mat <- as.matrix(theta1)
theta2Mat <- as.matrix(theta2)
inputLayerSize <- dim(theta1Mat)[2] - 1
hiddenLayerSize <- dim(theta2Mat)[2] - 1
numLabels <- dim(theta2Mat)[1]
theta1MatStack <- stack(as.data.frame(theta1Mat))
theta2MatStack <- stack(as.data.frame(theta2Mat))
theta1MatStackVals <- theta1MatStack[, "values"]
theta2MatStackVals <- theta2MatStack[, "values"]
thetaStack <- as.matrix(c(t(theta1MatStackVals), t(theta2MatStackVals)))

# Run feedforward section of neural network
print(sprintf("Feedforward Using Neural Network ..."))
lambda <- 0
neuralNetworkCost <- ComputeCost(thetaStack, xMat, yVec, lambda, 
                                 inputLayerSize, hiddenLayerSize, numLabels)
print(sprintf("Cost at parameters (loaded from Theta1.txt and Theta2.txt): 
               %.6f", neuralNetworkCost))
print(sprintf("(this value should be about 0.287629)"))
returnCode <- ReadKey()

# Run feedforward section of neural network with regularization
print(sprintf("Checking Cost Function (w/ Regularization) ..."))
lambda <- 1
neuralNetworkCost <- ComputeCost(thetaStack, xMat, yVec, lambda, 
                                 inputLayerSize, hiddenLayerSize, numLabels)
print(sprintf("Cost at parameters (loaded from Theta1.txt and Theta2.txt):
               %.6f", neuralNetworkCost))
print(sprintf("(this value should be about 0.383770)"))
returnCode <- ReadKey()

# Compute gradient for sigmoid function
print(sprintf("Evaluating sigmoid gradient..."))
sigmoidGradient <- ComputeSigmoidGradient(c(1, -0.5, 0, 0.5, 1))
print(sprintf("Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:"))
print(sprintf("%.6f", sigmoidGradient))
returnCode <- ReadKey()

# Train neural network
print(sprintf("Training Neural Network..."))
initTheta1 <- RandInitializeWeights(inputLayerSize, hiddenLayerSize)
initTheta2 <- RandInitializeWeights(hiddenLayerSize, numLabels)
initTheta1Stack <- stack(as.data.frame(initTheta1))
initTheta2Stack <- stack(as.data.frame(initTheta2))
initTheta1StackVals <- initTheta1Stack[, "values"]
initTheta2StackVals <- initTheta2Stack[, "values"]
initTheta <- as.matrix(c(t(initTheta1StackVals), t(initTheta2StackVals)))
optimResult <- optim(initTheta, fn=ComputeCost, gr=ComputeGradient, xMat, yVec,
                     lambda, inputLayerSize, hiddenLayerSize, numLabels, 
                     method="BFGS", control=list(maxit=75, trace=TRUE, 
                                                 REPORT=1))
theta1Mat <- matrix(optimResult$par[1:(hiddenLayerSize * 
                                       (inputLayerSize + 1)), ], 
                    nrow=hiddenLayerSize)
theta2Mat <- matrix(optimResult$par[(1 + hiddenLayerSize * 
                                     (inputLayerSize + 1)):
                                    (hiddenLayerSize * (inputLayerSize + 1) + 
                                     numLabels * (hiddenLayerSize + 1)), ], 
                    nrow=numLabels)
returnCode <- ReadKey()

# Visualize neural network
print(sprintf("Visualizing Neural Network..."))
returnCode <- DisplayData(theta1Mat[, -1])
returnCode <- ReadKey()

# Perform classification using trained neural network parameters
trainingPredict <- (Predict(theta1Mat, theta2Mat, xMat))
print(sprintf("Training Set Accuracy: %.6f",
              100 * apply((trainingPredict == yVec), 2, mean)))
