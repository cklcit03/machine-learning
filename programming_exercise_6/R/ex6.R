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
# Programming Exercise 6: Support Vector Machines
# Problem: Use SVMs to learn decision boundaries for various example datasets

# Load packages
library(e1071)

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
       xlab="", ylab="")
  points(x=negativeExamples[, 1], y=negativeExamples[, 2], col="black",
         bg="yellow", cex=1.75, pch=21, xlab="", ylab="")
  return(0)
}

VisualizeBoundaryLinear <- function(X, y, model) {
  # Plots linear decision boundary learned via SVM.
  #
  # Args:
  #   X: Features to be plotted.
  #   y: Data labels.
  #   model: Fitted SVM model.
  #
  # Returns:
  #   None.
  returnCode <- PlotData(X, y)
  weightVec <- t(model$coefs) %*% model$SV
  offsetVal <- -model$rho
  xVals <- seq(min(X[, 1]), max(X[, 1]), length=100)
  yLineVals <- (-1) * (weightVec[, 1] * xVals + offsetVal) / weightVec[, 2]
  lines(xVals, yLineVals, col="blue", pch="-")
  return(0)
}

VisualizeBoundary <- function(X, y, model) {
  # Plots non-linear decision boundary learned via SVM.
  #
  # Args:
  #   X: Features to be plotted.
  #   y: Data labels.
  #   model: Fitted SVM model.
  #
  # Returns:
  #   None.
  returnCode <- PlotData(X, y)
  x1 <- seq(min(X[, 1]), max(X[, 1]), length=100)
  x2 <- seq(min(X[, 2]), max(X[, 2]), length=100)
  valsMat <- mat.or.vec(dim(cbind(x1))[1], dim(cbind(x2))[1])
  for (index0 in 1:dim(cbind(x1))[1]) {
    for (index1 in 1:dim(cbind(x2))[1]) {
      valsMat[index0, index1] <- predict(model, cbind(x1[index0], x2[index1]))
    }
  }
  contour(x=x1, y=x2, valsMat, nlevels=1, col="blue", lwd=2, drawlabels=FALSE,
          add=TRUE)
  return(0)
}

Dataset3Params <- function(X1, y1, Xval, yVal) {
  # Selects optimal learning parameters for radial basis SVM.
  #
  # Args:
  #   X1: Matrix of training features.
  #   y1: Vector of training labels.
  #   Xval: Matrix of cross-validation features.
  #   yval: Vector of cross-validation labels.
  #
  # Returns:
  #   returnList: List of two objects.
  #               C: Best (in terms of minimum prediction error for 
  #                  cross-validation data) value that controls penalty for 
  #                  misclassified training examples.
  #               sigma: Best (in terms of minimum prediction error for 
  #                      cross-validation data) value that determines how fast
  #                      similarity metric decreases as examples are further 
  #                      apart.
  C <- 1
  sigma <- 0.3
  cArr <- cbind(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
  sigmaArr <- cbind(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
  bestPredErr <- 1000000
  for (cIndex in 1:dim(cArr)[2]) {
    for (sigmaIndex in 1:dim(sigmaArr)[2]) {
      svmModelTmp <- svm(X1, y=y1, scale=FALSE, type="C-classification", 
                         kernel="radial", 
                         gamma=1 / (2 * (sigmaArr[, sigmaIndex] ^ 2)), 
                         cost=cArr[, cIndex])
      predVec <- predict(svmModelTmp, Xval)
      currPredErr <- apply((predVec != yVal), 2, mean)
      if (currPredErr < bestPredErr) {
        cBest <- cArr[, cIndex]
        sigmaBest <- sigmaArr[, sigmaIndex]
        bestPredErr <- currPredErr
      }
    }
  }
  C <- cBest
  sigma <- sigmaBest
  returnList <- list("C"=C, "sigma"=sigma)
  return(returnList)
}

# Use setwd() to set working directory to directory that contains this source 
# file
# Load file into R
print(sprintf("Loading and Visualizing Data ..."))
exampleData1 <- read.csv("../exampleData1.txt", header=FALSE)
xMat <- cbind(exampleData1[, "V1"], exampleData1[, "V2"])
yVec <- cbind(exampleData1[, "V3"])

# Plot data
returnCode <- PlotData(cbind(exampleData1[, "V1"], exampleData1[, "V2"]),
                       exampleData1[, "V3"])
returnCode <- ReadKey()

# Train linear SVM on data
print(sprintf("Training Linear SVM ..."))
svmModel <- svm(xMat, y=yVec, scale=FALSE, type="C-classification", 
                kernel="linear")
returnCode <- VisualizeBoundaryLinear(xMat, yVec, svmModel)
returnCode <- ReadKey()

# Load next file into R
print(sprintf("Loading and Visualizing Data ..."))
exampleData2 <- read.csv("../exampleData2.txt", header=FALSE)
xMat2 <- cbind(exampleData2[, "V1"], exampleData2[, "V2"])
yVec2 <- cbind(exampleData2[, "V3"])

# Plot data
returnCode <- PlotData(cbind(exampleData2[, "V1"], exampleData2[, "V2"]),
                       exampleData2[, "V3"])
returnCode <- ReadKey()

# Train radial basis SVM on data
print(sprintf("Training SVM with RBF Kernel (this may take 1 to 2 minutes) 
              ..."))
kSigmaVal <- 0.1
svmModel2 <- svm(xMat2, y=yVec2, scale=FALSE, type="C-classification", 
                 kernel="radial", gamma=1 / (2 * (kSigmaVal ^ 2)))
returnCode <- VisualizeBoundary(xMat2, yVec2, svmModel2)
returnCode <- ReadKey()

# Load next file into R
print(sprintf("Loading and Visualizing Data ..."))
exampleData3 <- read.csv("../exampleData3.txt", header=FALSE)
exampleValData3 <- read.csv("../exampleValData3.txt", header=FALSE)
xMat3 <- cbind(exampleData3[, "V1"], exampleData3[, "V2"])
yVec3 <- cbind(exampleData3[, "V3"])
xValMat3 <- cbind(exampleValData3[, "V1"], exampleValData3[, "V2"])
yValVec3 <- cbind(exampleValData3[, "V3"])

# Plot data
returnCode <- PlotData(cbind(exampleData3[, "V1"], exampleData3[, "V2"]),
                       exampleData3[, "V3"])
returnCode <- ReadKey()

# Train radial basis SVM on data
dataset3ParamsList <- Dataset3Params(xMat3, yVec3, xValMat3, yValVec3)
svmModel3 <- svm(xMat3, y=yVec3, scale=FALSE, type="C-classification", 
                 kernel="radial", 
                 gamma=1 / (2 * (dataset3ParamsList$sigma ^ 2)), 
                 cost=dataset3ParamsList$C)
returnCode <- VisualizeBoundary(xMat3, yVec3, svmModel3)
returnCode <- ReadKey()
