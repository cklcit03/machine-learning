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
# Problem: Predict chances of university admission for an applicant given data
# for admissions decisions and test scores of various applicants

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
       xlab="", ylab="", xlim=c(30, 100), ylim=c(30, 100))
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
  returnCode <- PlotData(cbind(X[, 2], X[, 3]), y)
  yLineVals <- (theta[1] + theta[2] * X[, 2]) / (-1 * theta[3])
  lines(cbind(X[, 2]), yLineVals, col="blue", pch="-")
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
  cat("Program paused.  Press enter to continue.")
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

ComputeCost <- function(theta, X, y, numTrainEx) {
  # Computes cost function J(\theta).
  #
  # Args:
  #   theta: Vector of parameters for logistic regression.
  #   X: Matrix of features.
  #   y: Vector of labels.
  #   numTrainEx: Number of training examples.
  #
  # Returns:
  #   jTheta: Logistic regression cost.
  if (numTrainEx > 0) {
    hTheta <- ComputeSigmoid(X %*% theta)
    jTheta <- (colSums(-y * log(hTheta) 
                       - (1 - y) * log(1 - hTheta))) / numTrainEx
  } else {
    stop('Insufficient training examples')
  }
  return(jTheta)  
}

ComputeGradient <- function(theta, X, y, numTrainEx) {
  # Computes gradient of cost function J(\theta).
  #
  # Args:
  #   theta: Vector of parameters for logistic regression.
  #   X: Matrix of features.
  #   y: Vector of labels.
  #   numTrainEx: Number of training examples.
  #
  # Returns:
  #   gradArray: Vector of logistic regression gradients (one per feature).
  numFeatures <- dim(X)[2]
  if (numFeatures > 0) {
    if (numTrainEx > 0) {
      hTheta <- ComputeSigmoid(X %*% theta)
      gradArray <- matrix(0, numFeatures, 1)
      gradTermArray <- matrix(0, numTrainEx, numFeatures)
      for (gradIndex in 1:numFeatures) {
        gradTermArray[, gradIndex] <- (hTheta - y) * X[, gradIndex]
        gradArray[gradIndex] <- 
          (sum(gradTermArray[, gradIndex])) / (numTrainEx)
      }
    } else {
      stop('Insufficient training examples')
    }
  } else {
    stop('Insufficient features')
  }
  return(gradArray)
}

ComputeCostGradList <- function(X, y, theta) {
  # Aggregates computed cost and gradient.
  #
  # Args:
  #   X: Matrix of features.
  #   y: Vector of labels.
  #   theta: Vector of parameters for logistic regression.
  #
  # Returns:
  #   returnList: List of two objects.
  #               jTheta: Updated vector of parameters for logistic regression.
  #               gradArray: Updated vector of logistic regression gradients 
  #                          (one per feature).
  numTrainEx <- dim(y)[1]
  jTheta <- ComputeCost(theta, X, y, numTrainEx)
  gradArray <- ComputeGradient(theta, X, y, numTrainEx)
  returnList <- list("jTheta"=jTheta, "gradArray"=gradArray)
  return(returnList)
}

LabelPrediction <- function(X, theta) {
  # Performs label prediction on training data.
  #
  # Args:
  #   X: Matrix of features.
  #   theta: Vector of parameters for logistic regression.
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
applicantData <- read.csv("../applicantData.txt", header=FALSE)

# Plot data
print("Plotting data with + indicating (y = 1) examples and o indicating 
      (y = 0) examples.")
returnCode <- PlotData(cbind(applicantData[, "V1"], applicantData[, "V2"]),
                       applicantData[, "V3"])
title(xlab="Exam 1 score", ylab="Exam 2 score")
plotLegend <- legend('bottomright', col=c("black", "yellow"), 
                     pt.bg=c("black", "yellow"), pch=c(43, 22), pt.cex=1.75, 
                     legend=c("", ""), bty="n", trace=TRUE)
text(plotLegend$text$x - 3, plotLegend$text$y, c('Admitted', 'Not admitted'),
     pos=2)
returnCode <- ReadKey()
numTrainEx <- dim(applicantData)[1]
numFeatures <- dim(applicantData)[2] - 1
onesVec <- t(t(rep(1, numTrainEx)))
xMat <- cbind(onesVec, applicantData[, "V1"], applicantData[, "V2"])
yVec <- cbind(applicantData[, "V3"])
thetaVec <- t(t(rep(0, numFeatures + 1)))

# Compute initial cost and gradient
initComputeCostList <- ComputeCostGradList(xMat, yVec, thetaVec)
print(sprintf("Cost at initial theta (zeros): %.6f", 
              initComputeCostList$jTheta))
print(sprintf("Gradient at initial theta (zeros): "))
cat(format(round(initComputeCostList$gradArray, 6), nsmall=6), sep="\n")
returnCode <- ReadKey()

# Use optim to solve for optimum theta and cost
optimResult <- optim(thetaVec, fn=ComputeCost, gr=ComputeGradient, xMat, yVec,
                     numTrainEx, method="BFGS", control=list(maxit=400))
print(sprintf("Cost at theta found by optim: %.6f", optimResult$value))
print(sprintf("theta: "))
cat(format(round(optimResult$par, 6), nsmall=6), sep="\n")
returnCode <- PlotDecisionBoundary(xMat, yVec, optimResult$par)
title(xlab="Exam 1 score", ylab="Exam 2 score")
plotLegend <- legend('bottomright', col=c("black", "yellow", "blue"),
                     pt.bg=c("black", "yellow", "blue"), pch=c(43, 22, 45),
                     pt.cex=1.75, legend=c("", "", ""), bty="n", trace=TRUE)
text(plotLegend$text$x - 3, plotLegend$text$y, 
     c('Admitted', 'Not admitted', 'Decision Boundary'), pos=2)
returnCode <- ReadKey()

# Predict admission probability for a student with score 45 on exam 1 and score
# 85 on exam 2
admissionProb <- ComputeSigmoid(cbind(1, 45, 85) %*% optimResult$par)
print(sprintf("For a student with scores 45 and 85, we predict an admission 
              probability of %.6f", admissionProb))

# Compute accuracy on training set
trainingPredict <- (LabelPrediction(xMat, optimResult$par) + 0)
print(sprintf("Train Accuracy: %.6f",
              100 * apply((trainingPredict == yVec), 2, mean)))
returnCode <- ReadKey()
