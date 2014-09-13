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

# Display 2D data in a grid
displayData <- function(X){
  exampleWidth = round(sqrt(dim(X)[2]))
  numRows = dim(X)[1]
  numCols = dim(X)[2]
  exampleHeight = (numCols/exampleWidth)
  displayRows = floor(sqrt(numRows))
  displayCols = ceiling(numRows/displayRows)
  pad = 1
  displayArray = matrix(-1,pad+displayRows*(exampleHeight+pad),pad+displayCols*(exampleWidth+pad))
  currEx = 1
  for(rowIndex in 1:displayRows) {
    for(colIndex in 1:displayCols) {
      if (currEx > numRows) {
        break
      }
      maxVal = max(abs(X[currEx,]))
      rowIdx = pad+(rowIndex-1)*(exampleHeight+pad)+1:exampleHeight
      colIdx = pad+(colIndex-1)*(exampleWidth+pad)+1:exampleWidth
      xReshape = matrix(X[currEx,],nrow=exampleHeight,byrow=FALSE)
      displayArray[rowIdx,colIdx] = (1/maxVal)*as.numeric(t(xReshape[nrow(xReshape):1,]))
      currEx = currEx+1
    }
    if (currEx > numRows) {
      break
    }
  }
  image(displayArray,col=gray.colors(256),axes=FALSE,zlim=c(-1,1))
  return(0)
}

# Read key press
readKey <- function(){
  cat("Program paused. Press enter to continue.")
  line <- readline()
  return(0)
}

# Compute sigmoid function
computeSigmoid <- function(z){
  sigmoidZ = 1/(1+exp(-z))
  return(sigmoidZ)
}

# Compute regularized cost function J(\theta)
computeCost <- function(theta,X,y,numTrainEx,lambda){
  hTheta <- computeSigmoid(X%*%theta)
  thetaSquared = theta^2
  if (numTrainEx > 0)
    jTheta = (colSums(-y*log(hTheta)-(1-y)*log(1-hTheta)))/numTrainEx
  else
    stop('Insufficient training examples')
  jThetaReg = jTheta+(lambda/(2*numTrainEx))*sum(thetaSquared[-1])
  return(jThetaReg)
}

# Compute gradient of regularized cost function J(\theta)
computeGradient <- function(theta,X,y,numTrainEx,lambda){
  numFeatures = dim(X)[2]
  hTheta <- computeSigmoid(X%*%theta)
  gradArray = matrix(0,numFeatures,1)
  gradArrayReg = matrix(0,numFeatures,1)
  gradTermArray = matrix(0,numTrainEx,numFeatures)
  if (numFeatures > 0) {
    if (numTrainEx > 0) {
      for(gradIndex in 1:numFeatures) {
        gradTermArray[,gradIndex] = (hTheta-y)*X[,gradIndex]
        gradArray[gradIndex] = (sum(gradTermArray[,gradIndex]))/(numTrainEx)
        gradArrayReg[gradIndex] = gradArray[gradIndex]+(lambda/numTrainEx)*theta[gradIndex]
      }
      gradArrayReg[1] = gradArrayReg[1]-(lambda/numTrainEx)*theta[1]
    }
    else
      stop('Insufficient training examples')
  }
  else
    stop('Insufficient features')
  return(gradArrayReg)
}

# Train multiple logistic regression classifiers
oneVsAll <- function(X,y,numLabels,lambda){
  numTrainEx = dim(X)[1]
  numFeatures = dim(X)[2]
  allTheta = matrix(0,numLabels,numFeatures+1)
  onesVec = t(t(rep(1,numTrainEx)))
  augX = cbind(onesVec,X)
  for(labelIndex in 1:numLabels) {
    thetaVec = t(t(rep(0,numFeatures+1)))
    optimResult <- optim(thetaVec,fn=computeCost,gr=computeGradient,augX,as.numeric(yVec == labelIndex),numTrainEx,lambda,method="BFGS",control=list(maxit=400))
    allTheta[labelIndex,] = optimResult$par
  }
  return(allTheta)
}

# Perform label prediction on training data
predictOneVsAll <- function(X,allTheta){
  numTrainEx = dim(X)[1]
  onesVec = t(t(rep(1,numTrainEx)))
  augX = cbind(onesVec,X)
  sigmoidArr <- computeSigmoid(augX%*%t(allTheta))
  p = apply(sigmoidArr,1,which.max)
  return(p)
}

# Use setwd() to set working directory to directory that contains this source file
# Load file into R
print(sprintf("Loading and Visualizing Data ..."))
digitData = read.csv("../digitData.txt",header=FALSE)
numTrainEx = dim(digitData)[1]
xMat = as.matrix(subset(digitData,select=-c(V401)))
yVec = as.vector(subset(digitData,select=c(V401)))

# Randomly select 100 data points to display
randIndices = randperm(numTrainEx,numTrainEx)
xMatSel = subset(xMat,(rownames(xMat)) %in% randIndices[1])
for(randIndex in 2:100) {
  xMatSel = rbind(xMatSel,subset(xMat,(rownames(xMat)) %in% randIndices[randIndex]))
}
returnCode <- displayData(xMatSel)
returnCode <- readKey()

# Train one logistic regression classifier for each digit
print(sprintf("Training One-vs-All Logistic Regression..."))
lambda = 0.1
numLabels = 10
allTheta <- oneVsAll(xMat,yVec,numLabels,lambda)
returnCode <- readKey()

# Perform one-versus-all classification using logistic regression
trainingPredict <- (predictOneVsAll(xMat,allTheta))
print(sprintf("Training Set Accuracy: %.6f",100*apply((trainingPredict == yVec),2,mean)))
