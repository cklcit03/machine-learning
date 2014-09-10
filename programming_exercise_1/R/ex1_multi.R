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
# Programming Exercise 1: Linear Regression
# Problem: Predict housing prices given sizes/bedrooms of various houses
# Linear regression with multiple variables

# Load packages
library(pracma)

# Read key press
readKey <- function(){
  cat("Program paused.  Press enter to continue.")
  line <- readline()
  return(0)
}

# Perform feature normalization
featureNormalize <- function(X){
  numTrainEx = dim(X)[1]
  numFeatures = dim(X)[2]
  xNormalized = X
  muVec = colMeans(X)
  if (numFeatures >= 1) {
    sigmaVec = t(rep(0,numFeatures))
    for(index in 1:numFeatures) {
      sigmaVec[index] <- sd(X[,index])
    }
    if (numTrainEx >= 1) {
      for(index in 1:numTrainEx) {
        xNormalized[index,] = (X[index,]-muVec)/sigmaVec
      }
    }
    else
      stop('Insufficient training examples')
  }
  else
    stop('Insufficient features')
  returnList = list("xNormalized"=xNormalized,"muVec"=muVec,"sigmaVec"=sigmaVec)
  return(returnList)
}

# Compute cost function J(\theta)
computeCostMulti <- function(X,y,theta){
  numTrainEx = dim(y)[1]
  diffVec = X %*% theta-y
  diffVecSq = diffVec * diffVec
  if (numTrainEx > 0)
    jTheta = (colSums(diffVecSq))/(2*numTrainEx)
  else
    stop('Insufficient training examples')
  return(jTheta)
}

# Run gradient descent
gradientDescentMulti <- function(X,y,theta,alpha,numiters){
  numTrainEx = dim(y)[1]
  numFeatures = dim(X)[2]
  jThetaArray = t(t(rep(0,numiters)))
  if (numTrainEx > 0) {
    if (numFeatures >= 2) {
      if (numiters >= 1) {
        for(thetaIndex in 1:numiters) {
          diffVec = X %*% theta-y
          diffVecTimesX = cbind(diffVec * X[,1])
          for(featureIndex in 2:numFeatures) {
            diffVecTimesX = cbind(diffVecTimesX,diffVec * X[,featureIndex])
          }
          thetaNew = theta-t(alpha %*% (1/numTrainEx) %*% t(colSums(diffVecTimesX)))
          jThetaArray[thetaIndex] <- computeCostMulti(X,y,thetaNew)
          theta = thetaNew
        }
      }
      else
        stop('Insufficient iterations')
    }
    else
      stop('Insufficient features')
  }
  else
    stop('Insufficient training examples')
  returnList = list("theta"=theta,"jHistory"=jThetaArray)
  return(returnList)
}

# Compute normal equations
normalEqn <- function(X,y){
  theta = pinv(t(X) %*% X) %*% t(X) %*% y
  return(theta)
}

# Use setwd() to set working directory to directory that contains this source file
# Load file into R
print("Loading data ...")
housingData = read.csv("../housingData.txt",header=FALSE)
xMat = cbind(housingData[,"V1"],housingData[,"V2"])
yVec = cbind(housingData[,"V3"])
print("First 10 examples from the dataset: ")
for(trainExIndex in 1:10){
  print(sprintf("x = [%s], y = %d",paste(xMat[trainExIndex,],collapse=" "),yVec[trainExIndex,]))
}
returnCode <- readKey()

# Perform feature normalization
print("Normalizing Features ...")
featureNormalizeList <- featureNormalize(xMat)
xMatNormalized = featureNormalizeList$xNormalized
muVec = featureNormalizeList$muVec
sigmaVec = featureNormalizeList$sigmaVec
numTrainEx = dim(housingData)[1]
onesVec = t(t(rep(1,numTrainEx)))
xMatAug = cbind(onesVec,xMat)
xMatNormalizedAug = cbind(onesVec,xMatNormalized)
thetaVec = t(t(rep(0,3)))
iterations = 400
alpha = 0.1

# Run gradient descent
print("Running gradient descent ...")
gradientDescentMultiList <- gradientDescentMulti(xMatNormalizedAug,yVec,thetaVec,alpha,iterations)
thetaFinal = gradientDescentMultiList$theta
jHistory = gradientDescentMultiList$jHistory
plot(1:iterations,jHistory,type="l",col="blue",lwd=2,xlab="Number of iterations",ylab="Cost J")
print(sprintf("Theta computed from gradient descent: "))
cat(format(round(thetaFinal,6),nsmall=6),sep="\n")

# Predict price for a 1650 square-foot house with 3 bedrooms
xMatNormalized1 = (matrix(c(1650,3),nrow=1,ncol=2)-muVec)/sigmaVec
xMatNormalized1Aug = cbind(1,xMatNormalized1)
predPrice1 = xMatNormalized1Aug %*% thetaFinal
print(sprintf("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):"))
print(sprintf("$%.6f",predPrice1))
returnCode <- readKey()

# Solve normal equations
print("Solving with normal equations...")
thetaNormal <- normalEqn(xMatAug,yVec)
print(sprintf("Theta computed from the normal equations: "))
cat(format(round(thetaNormal,6),nsmall=6),sep="\n")

# Use normal equations to predict price for a 1650 square-foot house with 3 bedrooms
xMat2 = matrix(c(1,1650,3),nrow=1,ncol=3)
predPrice2 = xMat2 %*% thetaNormal
print(sprintf("Predicted price of a 1650 sq-ft, 3 br house (using normal equations):"))
print(sprintf("$%.6f",predPrice2))
