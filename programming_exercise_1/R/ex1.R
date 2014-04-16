# Machine Learning
# Programming Exercise 1: Linear Regression
# Problem: Predict profits for a food truck given data for profits/populations of various cities
# Linear regression with one variable

# Compute cost function J(\theta)
computeCost <- function(X,y,theta){
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
gradientDescent <- function(X,y,theta,alpha,numiters){
  numTrainEx = dim(y)[1]
  jThetaArray = t(t(rep(numiters)))
  if (numTrainEx > 0) {
    if (numiters >= 1) {
      for(thetaIndex in 1:numiters) {
        diffVec = X %*% theta-y
        diffVecTimesX = cbind(diffVec*X[,1],diffVec*X[,2])
        thetaNew = theta-t(alpha %*% (1/numTrainEx) %*% t(colSums(diffVecTimesX)))
        jThetaArray[thetaIndex] <- computeCost(X,y,thetaNew)
        theta = thetaNew
      }
    }
    else
      theta = t(t(rep(0,2)))
  }
  else
    theta = t(t(rep(0,2)))
  return(theta)
}

# Use setwd() to set working directory to directory that contains this source file
# Load file into R
foodTruckData = read.csv("../foodTruckData.txt",header=FALSE)
numTrainEx = dim(foodTruckData)[1]
onesVec = t(t(rep(1,numTrainEx)))
xMat = cbind(onesVec,foodTruckData[,"V1"])
thetaVec = t(t(rep(0,2)))
yVec = cbind(foodTruckData[,"V2"])

# Compute initial cost
initCost <- computeCost(xMat,yVec,thetaVec)

# Run gradient descent
iterations = 1500
alpha = 0.01
thetaFinal <- gradientDescent(xMat,yVec,thetaVec,alpha,iterations)

# Predict profit for population size of 35000
predProfit1 = matrix(c(1,3.5),nrow=1,ncol=2) %*% thetaFinal
predProfit1Scaled = 10000*predProfit1

# Predict profit for population size of 70000
predProfit2 = matrix(c(1,7),nrow=1,ncol=2) %*% thetaFinal
predProfit2Scaled = 10000*predProfit2
