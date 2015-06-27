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
# Programming Exercise 5: Regularized Linear Regression and Bias vs. Variance
# Problem: Predict amount of water flowing out of a dam given data for 
# change of water level in a reservoir

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

ComputeCost <- function(theta, X, y, lambda) {
  # Computes regularized cost function J(\theta).
  #
  # Args:
  #   theta: Vector of parameters for regularized linear regression.
  #   X: Matrix of features.
  #   y: Vector of labels.
  #   lambda: Regularization parameter.
  #
  # Returns:
  #   jThetaReg: Regularized linear regression cost.
  numTrainEx <- dim(X)[1]
  if (numTrainEx > 0) {
    diffVec <- X %*% theta - y
    diffVecSq <- diffVec * diffVec
    jTheta <- (colSums(diffVecSq)) / (2 * numTrainEx)
    thetaSquared <- theta ^ 2
    jThetaReg <- jTheta + (lambda / (2 * numTrainEx)) * sum(thetaSquared[-1])
  } else {
    stop('Insufficient training examples')
  }
  return(jThetaReg)
}

ComputeGradient <- function(theta, X, y, lambda) {
  # Computes gradient of regularized cost function J(\theta).
  #
  # Args:
  #   theta: Vector of parameters for regularized linear regression.
  #   X: Matrix of features.
  #   y: Vector of labels.
  #   lambda: Regularization parameter.
  #
  # Returns:
  #   gradArrayReg: Vector of regularized linear regression gradients (one 
  #                 per feature).
  numTrainEx <- dim(X)[1]
  numFeatures <- dim(X)[2]
  if (numFeatures > 0) {
    if (numTrainEx > 0) {
      hTheta <- X %*% theta
      gradArray <- matrix(0, numFeatures, 1)
      gradArrayReg <- matrix(0, numFeatures, 1)
      gradTermArray <- matrix(0, numTrainEx, numFeatures)
      for (gradIndex in 1:numFeatures) {
        gradTermArray[, gradIndex] <- (hTheta - y) * X[, gradIndex]
        gradArray[gradIndex] <- (sum(gradTermArray[, gradIndex])) / numTrainEx
        gradArrayReg[gradIndex] <- gradArray[gradIndex] + 
          (lambda / numTrainEx) * theta[gradIndex]
        gradArrayReg[1] <- gradArrayReg[1] - (lambda / numTrainEx) * theta[1]
      }
    } else {
      stop('Insufficient training examples')
    }
  } else {
    stop('Insufficient features')
  }
  return(gradArrayReg)
}

TrainLinearReg <- function(X, y, lambda) {
  # Trains linear regression.
  #
  # Args:
  #   X: Matrix of features.
  #   y: Vector of labels.
  #   lambda: Regularization parameter.
  #
  # Returns:
  #   optimResult$par: Best set of parameters found by optim.
  numFeatures <- dim(X)[2]
  if (numFeatures > 0) {
    initTheta <- t(t(rep(1, numFeatures)))
    optimResult <- optim(initTheta, fn=ComputeCost, gr=ComputeGradient, X, y, 
                         lambda, method="BFGS", 
                         control=list(maxit=75, trace=TRUE, REPORT=1))
  } else {
    stop('Insufficient features')
  }
  return(optimResult$par)
}

LearningCurve <- function(X, y, XVal, yVal, lambda) {
  # Generates values for learning curve.
  #
  # Args:
  #   X: Matrix of training features.
  #   y: Vector of training labels.
  #   Xval: Matrix of cross-validation features.
  #   yVal: Vector of cross-validation labels.
  #   lambda: Regularization parameter.
  #
  # Returns:
  #   returnList: List of two objects.
  #               errorTrain: Vector of regularized linear regression costs for
  #                           training data (one per example).
  #               errorVal: Vector of regularized linear regression costs for
  #                         cross-validation data (one per example).
  numTrainEx <- dim(y)[1]
  if (numTrainEx > 0) {
    errorTrain <- t(t(rep(0, numTrainEx)))
    errorVal <- t(t(rep(0, numTrainEx)))
    for (exIndex in 1:numTrainEx) {
      if (exIndex == 1) {
        XSubMat <- t(as.matrix(X[1:exIndex, ]))
      } else {
        XSubMat <- as.matrix(X[1:exIndex, ])
      }
      ySubVec <- y[1:exIndex, ]
      trainThetaVec <- TrainLinearReg(XSubMat, ySubVec, 1)
      errorTrain[exIndex, ] <- ComputeCost(trainThetaVec, XSubMat, ySubVec, 0)
      errorVal[exIndex, ] <- ComputeCost(trainThetaVec, XVal, yVal, 0)
    }
  } else {
    stop('Insufficient training examples')
  }
  returnList <- list("errorTrain"=errorTrain, "errorVal"=errorVal)
  return(returnList)
}

ValidationCurve <- function(X, y, XVal, yVal) {
  # Generates values for validation curve.
  #
  # Args:
  #   X: Matrix of training features.
  #   y: Vector of training labels.
  #   Xval: Matrix of cross-validation features.
  #   yVal: Vector of cross-validation labels.
  #
  # Returns:
  #   returnList: List of three objects.
  #               lambdaVec: Vector of regularization parameters.
  #               errorTrain: Vector of regularized linear regression costs for
  #                           training data (one per regularization parameter).
  #               errorVal: Vector of regularized linear regression costs for
  #                         cross-validation data (one per regularization 
  #                         parameter).
  lambdaVec <- t(t(c(0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)))
  numLambda <- dim(lambdaVec)[1]
  errorTrain <- t(t(rep(0, numLambda)))
  errorVal <- t(t(rep(0, numLambda)))
  for (lambdaIndex in 1:numLambda) {
    currLambda <- lambdaVec[lambdaIndex, ]
    trainThetaVec <- TrainLinearReg(X, y, currLambda)
    errorTrain[lambdaIndex, ] <- ComputeCost(trainThetaVec, X, y, 0)
    errorVal[lambdaIndex, ] <- ComputeCost(trainThetaVec, XVal, yVal, 0)
  }
  returnList <- list("lambdaVec"=lambdaVec, "errorTrain"=errorTrain, 
                     "errorVal"=errorVal)
  return(returnList)
}

PolyFeatures <- function(X, p) {
  # Performs feature mapping for polynomial regression.
  #
  # Args:
  #   X: Matrix of training features.
  #   p: Maximum degree of polynomial mapping.
  #
  # Returns:
  #   XPoly: Matrix of mapped features.
  XPoly <- matrix(0, dim(X)[1], p)
  if (p > 0) {
    for (degIndex in 1:p) {
      XPoly[, degIndex] <- X ^ degIndex
    }
  } else {
    stop('Insufficient degree')
  }
  return(XPoly)
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

PlotFit <- function(minX, maxX, mu, sigma, theta, p) {
  # Plots polynomial regression fit.
  #
  # Args:
  #   minX: Lower bound for x-axis of plot.
  #   maxX: Upper bound for x-axis of plot.
  #   mu: Vector of mean values of mapped features.
  #   sigma: Vector of standard deviations of mapped features.
  #   theta: Vector of learned parameters for polynomial regression.
  #   p: Maximum degree of polynomial mapping.
  #
  # Returns:
  #   None.
  xSeq <- t(t(seq(minX - 15, maxX + 25, by=0.05)))
  xPoly <- PolyFeatures(xSeq, p)
  xPolyNorm <- matrix(0, dim(xPoly)[1], p)
  for (index in 1:dim(xPoly)[1]) {
    xPolyNorm[index, ] <- (xPoly[index, ] - mu) / sigma
  }
  onesVec <- t(t(rep(1, dim(xPoly)[1])))
  xPolyNorm <- cbind(onesVec, xPolyNorm)
  points(xSeq, xPolyNorm %*% theta, col="blue", pch='-', cex=1.25)
  return(0)
}

# Use setwd() to set working directory to directory that contains this source 
# file
# Load file into R
print(sprintf("Loading and Visualizing Data ..."))
waterTrainData <- read.csv("../waterTrainData.txt", header=FALSE)
numTrainEx <- dim(waterTrainData)[1]
waterValData <- read.csv("../waterValData.txt", header=FALSE)
numValEx <- dim(waterValData)[1]
waterTestData <- read.csv("../waterTestData.txt", header=FALSE)
numTestEx <- dim(waterTestData)[1]

# Plot training data
onesTrainVec <- t(t(rep(1, numTrainEx)))
xMat <- cbind(waterTrainData[, "V1"])
yVec <- cbind(waterTrainData[, "V2"])
plot(xMat, yVec, col="red", cex=1.75, pch="x", 
     xlab="Change in water level (x)", ylab="Water flowing out of the dam (y)")
returnCode <- ReadKey()

# Compute cost for regularized linear regression
xMat <- cbind(onesTrainVec, xMat)
thetaVec <- t(t(rep(1, 2)))
initCost <- ComputeCost(thetaVec, xMat, yVec, 1)
print(sprintf("Cost at theta = [1 ; 1]: %.6f", initCost))
print(sprintf("(this value should be about 303.993192)"))
returnCode <- ReadKey()

# Compute gradient for regularized linear regression
initGradient <- ComputeGradient(thetaVec, xMat, yVec, 1)
print(sprintf("Gradient at theta = [1 ; 1]: "))
cat(format(round(t(initGradient), 6), nsmall=6), sep=" ")
print(sprintf("(this value should be about [-15.303016; 598.250744])"))
returnCode <- ReadKey()

# Train linear regression
lambda <- 0
trainThetaVec <- TrainLinearReg(xMat, yVec, lambda)

# Plot fit over data
lines(waterTrainData[, "V1"], xMat %*% trainThetaVec, col="blue", pch="-")
returnCode <- ReadKey()

# Generate values for learning curve
onesValVec <- t(t(rep(1, numValEx)))
xValMat <- cbind(onesValVec, waterValData[, "V1"])
yValVec <- cbind(waterValData[, "V2"])
learningCurveList <- LearningCurve(xMat, yVec, xValMat, yValVec, lambda)

# Plot learning curve
plot(seq(numTrainEx), learningCurveList$errorTrain, type='l', col="blue",
     main="Learning curve for linear regression", 
     xlab="Number of training examples", ylab="Error", xlim=c(0, 13), 
     ylim=c(0, 150))
lines(seq(numTrainEx), learningCurveList$errorVal, col="green")
plotLegend <- legend('topright', legend=c("", ""), lty=c(1, 1), 
                     lwd=c(2.5, 2.5), col=c("blue", "green"), bty="n", 
                     trace=TRUE)
text(plotLegend$text$x - 1, plotLegend$text$y, c('Train', 'Cross Validation'),
     pos=2)
dispMat <- c(1, signif(learningCurveList$errorTrain[1, ], 6), 
             signif(learningCurveList$errorVal[1, ], 6))
for (exIdx in 2:numTrainEx) {
  dispMat <- rbind(dispMat, 
                   c(exIdx, signif(learningCurveList$errorTrain[exIdx, ], 6), 
                     signif(learningCurveList$errorVal[exIdx, ], 6)))
}
colnames(dispMat) <- c("# Training Examples", "Train Error", 
                       "Cross Validation Error")
dispMat <- as.table(dispMat)
dispMat
returnCode <- ReadKey()

# Perform feature mapping for polynomial regression
kP <- 8
xPoly <- PolyFeatures(t(t(xMat[, 2])), kP)
xPolyNorm <- FeatureNormalize(xPoly)
xPolyNorm$xNormalized <- cbind(onesTrainVec, xPolyNorm$xNormalized)
xTestMat <- cbind(waterTestData[, "V1"])
xTestPoly <- PolyFeatures(xTestMat, kP)
xTestPolyNorm <- matrix(0, numTestEx, kP)
for (index in 1:numTestEx) {
  xTestPolyNorm[index, ] <- 
    (xTestPoly[index, ] - xPolyNorm$muVec) / xPolyNorm$sigmaVec
}
onesTestVec <- t(t(rep(1, numTestEx)))
xTestPolyNorm <- cbind(onesTestVec, xTestPolyNorm)
xValPoly <- PolyFeatures(as.matrix(xValMat[, 2]), kP)
xValPolyNorm <- matrix(0, numValEx, kP)
for (index in 1:numValEx) {
  xValPolyNorm[index, ] <- 
    (xValPoly[index, ] - xPolyNorm$muVec) / xPolyNorm$sigmaVec
}
xValPolyNorm <- cbind(onesValVec, xValPolyNorm)
print(sprintf("Normalized Training Example 1:"))
cat(format(round(xPolyNorm$xNormalized[1, ], 6), nsmall=6), sep="\n")
returnCode <- ReadKey()

# Train polynomial regression
lambda <- 0
trainThetaVec <- TrainLinearReg(xPolyNorm$xNormalized, yVec, lambda)

# Plot fit over data
plot(xMat[, 2], yVec, col="red", pch="x", cex=1.75, xlim=c(-100, 100), 
     ylim=c(-60, 40), main=sprintf("Polynomial Regression Fit (lambda = %f)", 
                                   lambda), xlab="Change in water level (x)",
     ylab="Water flowing out of the dam (y)")
returnCode <- PlotFit(min(xMat[, 2]), max(xMat[, 2]), xPolyNorm$muVec, 
                      xPolyNorm$sigmaVec, trainThetaVec, kP)

# Generate values for learning curve for polynomial regression
learningCurveList <- LearningCurve(xPolyNorm$xNormalized, yVec, xValPolyNorm, 
                                   yValVec, lambda)

# Plot learning curve
plot(seq(numTrainEx), learningCurveList$errorTrain, type='l', col="blue", 
     main=sprintf("Polynomial Regression Learning Curve (lambda = %f)", 
                  lambda), xlab="Number of training examples", ylab="Error", 
     xlim=c(0, 13), ylim=c(0, 100))
lines(seq(numTrainEx), learningCurveList$errorVal, col="green")
plotLegend <- legend('topright', legend=c("", ""), lty=c(1, 1), 
                     lwd=c(2.5, 2.5), col=c("blue", "green"), bty="n", 
                     trace=TRUE)
text(plotLegend$text$x - 1, plotLegend$text$y, c('Train', 'Cross Validation'), 
     pos=2)
print(sprintf("Polynomial Regression (lambda = %f)", lambda))
dispMat <- c(1, signif(learningCurveList$errorTrain[1, ], 6), 
             signif(learningCurveList$errorVal[1, ], 6))
for (exIdx in 2:numTrainEx) {
  dispMat <- rbind(dispMat,
                   c(exIdx, signif(learningCurveList$errorTrain[exIdx, ], 6), 
                     signif(learningCurveList$errorVal[exIdx, ], 6)))
}
colnames(dispMat) <- c("# Training Examples", "Train Error", 
                       "Cross Validation Error")
dispMat <- as.table(dispMat)
dispMat
returnCode <- ReadKey()

# Generate values for validation curve for polynomial regression
validationCurveList <- ValidationCurve(xPolyNorm$xNormalized, yVec, 
                                       xValPolyNorm, yValVec)

# Plot validation curve
plot(validationCurveList$lambdaVec, validationCurveList$errorTrain, type='l', 
     col="blue", xlab="lambda", ylab="Error")
lines(validationCurveList$lambdaVec, validationCurveList$errorVal, col="green")
plotLegend <- legend('bottomright', legend=c("", ""), lty=c(1, 1), 
                     lwd=c(2.5, 2.5), col=c("blue", "green"), bty="n", 
                     trace=TRUE)
text(plotLegend$text$x - 1, plotLegend$text$y, c('Train', 'Cross Validation'), 
     pos=2)
dispMat <- c(validationCurveList$lambdaVec[1, ], 
             signif(validationCurveList$errorTrain[1, ], 6), 
             signif(validationCurveList$errorVal[1, ], 6))
for (lambdaIndex in 2:dim(validationCurveList$lambdaVec)[1]) {
  dispMat <- rbind(dispMat, 
                   c(validationCurveList$lambdaVec[lambdaIndex, ], 
                     signif(validationCurveList$errorTrain[lambdaIndex, ], 6), 
                     signif(validationCurveList$errorVal[lambdaIndex, ], 6)))
}
colnames(dispMat) <- c("lambda", "Train Error", "Validation Error")
dispMat <- as.table(dispMat)
dispMat
returnCode <- ReadKey()
