# Machine Learning
# Programming Exercise 1: Linear Regression
# Problem: Predict housing prices given sizes/bedrooms of various houses
# Linear regression with multiple variables

# Load packages
library(pracma)

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
    else {
      xNormalized = 0
      muVec = t(rep(0,2))
      sigmaVec = t(rep(1,2))
    }
  }
  else {
    xNormalized = t(t(rep(0,numTrainEx)))
    muVec = t(rep(0,2))
    sigmaVec = t(rep(1,2))
  }
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
    jTheta = 0
  return(jTheta)
}

# Run gradient descent
gradientDescentMulti <- function(X,y,theta,alpha,numiters){
  numTrainEx = dim(y)[1]
  numFeatures = dim(X)[2]
  jThetaArray = t(t(rep(0,numTrainEx)))
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
        theta = t(t(rep(0,3)))
    }
    else
      theta = t(t(rep(0,3)))
  }
  else
    theta = t(t(rep(0,3)))
  return(theta)
}

# Compute normal equations
normalEqn <- function(X,y){
  theta = pinv(t(X) %*% X) %*% t(X) %*% y
  return(theta)
}

# Use setwd() to set working directory to directory that contains this source file
# Load file into R
housingData = read.csv("../housingData.txt",header=FALSE)
xMat = cbind(housingData[,"V1"],housingData[,"V2"])

# Perform feature normalization
featureNormalizeList <- featureNormalize(xMat)
xMatNormalized = featureNormalizeList$xNormalized
muVec = featureNormalizeList$muVec
sigmaVec = featureNormalizeList$sigmaVec
numTrainEx = dim(housingData)[1]
onesVec = t(t(rep(1,numTrainEx)))
xMatAug = cbind(onesVec,xMat)
xMatNormalizedAug = cbind(onesVec,xMatNormalized)
yVec = cbind(housingData[,"V3"])
thetaVec = t(t(rep(0,3)))
iterations = 400
alpha = 0.1

# Run gradient descent
thetaFinal <- gradientDescentMulti(xMatNormalizedAug,yVec,thetaVec,alpha,iterations)

# Predict price for a 1650 square-foot house with 3 bedrooms
xMatNormalized1 = (matrix(c(1650,3),nrow=1,ncol=2)-muVec)/sigmaVec
xMatNormalized1Aug = cbind(1,xMatNormalized1)
predPrice1 = xMatNormalized1Aug %*% thetaFinal

# Solve normal equations
thetaNormal <- normalEqn(xMatAug,yVec)

# Use normal equations to predict price for a 1650 square-foot house with 3 bedrooms
xMat2 = matrix(c(1,1650,3),nrow=1,ncol=3)
predPrice2 = xMat2 %*% thetaNormal
