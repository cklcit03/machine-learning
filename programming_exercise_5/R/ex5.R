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

# Load packages
library(pracma)

# Read key press
readKey <- function(){
  cat("Program paused. Press enter to continue.")
  line <- readline()
  return(0)
}

# Compute regularized cost function J(\theta)
computeCost <- function(theta,X,y,lambda){
  numTrainEx = dim(X)[1]
  diffVec = X %*% theta-y
  diffVecSq = diffVec * diffVec
  if (numTrainEx > 0)
    jTheta = (colSums(diffVecSq))/(2*numTrainEx)
  else
    stop('Insufficient training examples')
  thetaSquared = theta^2
  jThetaReg = jTheta+(lambda/(2*numTrainEx))*sum(thetaSquared[-1])
  return(jThetaReg)
}

# Compute gradient of regularized cost function J(\theta)
computeGradient <- function(theta,X,y,lambda){
  numTrainEx = dim(X)[1]
  numFeatures = dim(X)[2]
  hTheta  = X%*%theta
  if (numFeatures > 0) {
    if (numTrainEx > 0) {
      gradArray = matrix(0,numFeatures,1)
      gradArrayReg = matrix(0,numFeatures,1)
      gradTermArray = matrix(0,numTrainEx,numFeatures)
      for(gradIndex in 1:numFeatures) {
        gradTermArray[,gradIndex] = (hTheta-y)*X[,gradIndex]
        gradArray[gradIndex] = (sum(gradTermArray[,gradIndex]))/(numTrainEx)
        gradArrayReg[gradIndex] = gradArray[gradIndex]+(lambda/numTrainEx)*theta[gradIndex]
        gradArrayReg[1] = gradArrayReg[1]-(lambda/numTrainEx)*theta[1]
      }
    }
    else
      stop('Insufficient training examples')
  }
  else
    stop('Insufficient features')
  return(gradArrayReg)
}

# Train linear regression
trainLinearReg <- function(X,y,lambda){
  numFeatures = dim(X)[2]
  if (numFeatures > 0) {
    initTheta = t(t(rep(1,numFeatures)))
    optimResult <- optim(initTheta,fn=computeCost,gr=computeGradient,X,y,lambda,method="BFGS",control=list(maxit=75,trace=TRUE,REPORT=1))
  }
  else
    stop('Insufficient features')
  return(optimResult$par)
}

# Generate values for learning curve
learningCurve <- function(X,y,XVal,yVal,lambda) {
  numTrainEx = dim(y)[1]
  errorTrain = t(t(rep(0,numTrainEx)))
  errorVal = t(t(rep(0,numTrainEx)))
  if (numTrainEx > 0) {
    for(exIndex in 1:numTrainEx) {
      if (exIndex == 1)
        XSubMat = t(as.matrix(X[1:exIndex,]))
      else
        XSubMat = as.matrix(X[1:exIndex,])
      ySubVec = y[1:exIndex,]
      trainThetaVec <- trainLinearReg(XSubMat,ySubVec,1)
      errorTrain[exIndex,] <- computeCost(trainThetaVec,XSubMat,ySubVec,0)
      errorVal[exIndex,] <- computeCost(trainThetaVec,XVal,yVal,0)
    }
  }
  else
    stop('Insufficient training examples')
  returnList = list("errorTrain"=errorTrain,"errorVal"=errorVal)
  return(returnList)
}

# Generate values for validation curve
validationCurve <- function(X,y,XVal,yVal) {
  lambdaVec = t(t(c(0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10)))
  numLambda = dim(lambdaVec)[1]
  errorTrain = t(t(rep(0,numLambda)))
  errorVal = t(t(rep(0,numLambda)))
  numTrainEx = dim(X)[1]
  for(lambdaIndex in 1:numLambda) {
    currLambda = lambdaVec[lambdaIndex,]
    trainThetaVec <- trainLinearReg(X,y,currLambda)
    errorTrain[lambdaIndex,] <- computeCost(trainThetaVec,X,y,0)
    errorVal[lambdaIndex,] <- computeCost(trainThetaVec,XVal,yVal,0)
  }
  returnList = list("lambdaVec"=lambdaVec,"errorTrain"=errorTrain,"errorVal"=errorVal)
  return(returnList)
}

# Perform feature mapping for polynomial regression
polyFeatures <- function(X,p) {
  XPoly = matrix(0,dim(X)[1],p)
  if (p > 0) {
    for(degIndex in 1:p) {
      XPoly[,degIndex] = X^degIndex
    }
  }
  else
    stop('Insufficient degree')
  return(XPoly)
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

# Plot polynomial regression fit
plotFit <- function(minX,maxX,mu,sigma,theta,p){
  xSeq = t(t(seq(minX-15,maxX+25,by=0.05)))
  xPoly = polyFeatures(xSeq,p)
  xPolyNorm = matrix(0,dim(xPoly)[1],p)
  for(index in 1:dim(xPoly)[1]) {
    xPolyNorm[index,] = (xPoly[index,]-mu)/sigma
  }
  onesVec = t(t(rep(1,dim(xPoly)[1])))
  xPolyNorm = cbind(onesVec,xPolyNorm)
  points(xSeq,xPolyNorm%*%theta,col="blue",pch='-',cex=1.25)
  return(0)
}

# Use setwd() to set working directory to directory that contains this source file
# Load file into R
print(sprintf("Loading and Visualizing Data ..."))
waterTrainData = read.csv("../waterTrainData.txt",header=FALSE)
numTrainEx = dim(waterTrainData)[1]
waterValData = read.csv("../waterValData.txt",header=FALSE)
numValEx = dim(waterValData)[1]
waterTestData = read.csv("../waterTestData.txt",header=FALSE)
numTestEx = dim(waterTestData)[1]

# Plot training data
onesTrainVec = t(t(rep(1,numTrainEx)))
xMat = cbind(waterTrainData[,"V1"])
yVec = cbind(waterTrainData[,"V2"])
plot(xMat,yVec,col="red",cex=1.75,pch="x",xlab="Change in water level (x)",ylab="Water flowing out of the dam (y)")
returnCode <- readKey()

# Compute cost for regularized linear regression
xMat = cbind(onesTrainVec,xMat)
thetaVec = t(t(rep(1,2)))
initCost <- computeCost(thetaVec,xMat,yVec,1)
print(sprintf("Cost at theta = [1 ; 1]: %.6f",initCost))
print(sprintf("(this value should be about 303.993192)"))
returnCode <- readKey()

# Compute gradient for regularized linear regression
initGradient <- computeGradient(thetaVec,xMat,yVec,1)
print(sprintf("Gradient at theta = [1 ; 1]: "))
cat(format(round(t(initGradient),6),nsmall=6),sep=" ")
print(sprintf("(this value should be about [-15.303016; 598.250744])"))
returnCode <- readKey()

# Train linear regression
lambda = 0
trainThetaVec <- trainLinearReg(xMat,yVec,lambda)

# Plot fit over data
lines(waterTrainData[,"V1"],xMat%*%trainThetaVec,col="blue",pch="-")
returnCode <- readKey()

# Generate values for learning curve
onesValVec = t(t(rep(1,numValEx)))
xValMat = cbind(onesValVec,waterValData[,"V1"])
yValVec = cbind(waterValData[,"V2"])
learningCurveList <- learningCurve(xMat,yVec,xValMat,yValVec,lambda)

# Plot learning curve
plot(seq(numTrainEx),learningCurveList$errorTrain,type='l',col="blue",main="Learning curve for linear regression",xlab="Number of training examples",ylab="Error",xlim=c(0,13),ylim=c(0,150))
lines(seq(numTrainEx),learningCurveList$errorVal,col="green")
plotLegend <- legend('topright',legend=c("",""),lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","green"),bty="n",trace=TRUE)
text(plotLegend$text$x-1,plotLegend$text$y,c('Train','Cross Validation'),pos=2)
dispMat = c(1,signif(learningCurveList$errorTrain[1,],6),signif(learningCurveList$errorVal[1,],6))
for(exIndex in 2:numTrainEx) {
  dispMat = rbind(dispMat,c(exIndex,signif(learningCurveList$errorTrain[exIndex,],6),signif(learningCurveList$errorVal[exIndex,],6)))
}
colnames(dispMat) = c("# Training Examples","Train Error","Cross Validation Error")
dispMat = as.table(dispMat)
dispMat
returnCode <- readKey()

# Perform feature mapping for polynomial regression
p = 8
xPoly <- polyFeatures(t(t(xMat[,2])),p)
xPolyNorm <- featureNormalize(xPoly)
xPolyNorm$xNormalized = cbind(onesTrainVec,xPolyNorm$xNormalized)
xTestMat = cbind(waterTestData[,"V1"])
xTestPoly <- polyFeatures(xTestMat,p)
xTestPolyNorm = matrix(0,numTestEx,p)
for(index in 1:numTestEx) {
  xTestPolyNorm[index,] = (xTestPoly[index,]-xPolyNorm$muVec)/xPolyNorm$sigmaVec
}
onesTestVec = t(t(rep(1,numTestEx)))
xTestPolyNorm = cbind(onesTestVec,xTestPolyNorm)
xValPoly <- polyFeatures(as.matrix(xValMat[,2]),p)
xValPolyNorm = matrix(0,numValEx,p)
for(index in 1:numValEx) {
  xValPolyNorm[index,] = (xValPoly[index,]-xPolyNorm$muVec)/xPolyNorm$sigmaVec
}
xValPolyNorm = cbind(onesValVec,xValPolyNorm)
print(sprintf("Normalized Training Example 1:"))
cat(format(round(xPolyNorm$xNormalized[1,],6),nsmall=6),sep="\n")
returnCode <- readKey()

# Train polynomial regression
lambda = 0
trainThetaVec <- trainLinearReg(xPolyNorm$xNormalized,yVec,lambda)

# Plot fit over data
plot(xMat[,2],yVec,col="red",pch="x",cex=1.75,xlim=c(-100,100),ylim=c(-60,40),main=sprintf("Polynomial Regression Fit (lambda = %f)",lambda),xlab="Change in water level (x)",ylab="Water flowing out of the dam (y)")
returnCode <- plotFit(min(xMat[,2]),max(xMat[,2]),xPolyNorm$muVec,xPolyNorm$sigmaVec,trainThetaVec,p)

# Generate values for learning curve for polynomial regression
learningCurveList <- learningCurve(xPolyNorm$xNormalized,yVec,xValPolyNorm,yValVec,lambda)

# Plot learning curve
plot(seq(numTrainEx),learningCurveList$errorTrain,type='l',col="blue",main=sprintf("Polynomial Regression Learning Curve (lambda = %f)",lambda),xlab="Number of training examples",ylab="Error",xlim=c(0,13),ylim=c(0,100))
lines(seq(numTrainEx),learningCurveList$errorVal,col="green")
plotLegend <- legend('topright',legend=c("",""),lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","green"),bty="n",trace=TRUE)
text(plotLegend$text$x-1,plotLegend$text$y,c('Train','Cross Validation'),pos=2)
print(sprintf("Polynomial Regression (lambda = %f)",lambda))
dispMat = c(1,signif(learningCurveList$errorTrain[1,],6),signif(learningCurveList$errorVal[1,],6))
for(exIndex in 2:numTrainEx) {
  dispMat = rbind(dispMat,c(exIndex,signif(learningCurveList$errorTrain[exIndex,],6),signif(learningCurveList$errorVal[exIndex,],6)))
}
colnames(dispMat) = c("# Training Examples","Train Error","Cross Validation Error")
dispMat = as.table(dispMat)
dispMat
returnCode <- readKey()

# Generate values for validation curve for polynomial regression
validationCurveList <- validationCurve(xPolyNorm$xNormalized,yVec,xValPolyNorm,yValVec)

# Plot validation curve
plot(validationCurveList$lambdaVec,validationCurveList$errorTrain,type='l',col="blue",xlab="lambda",ylab="Error")
lines(validationCurveList$lambdaVec,validationCurveList$errorVal,col="green")
plotLegend <- legend('bottomright',legend=c("",""),lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","green"),bty="n",trace=TRUE)
text(plotLegend$text$x-1,plotLegend$text$y,c('Train','Cross Validation'),pos=2)
dispMat = c(validationCurveList$lambdaVec[1,],signif(validationCurveList$errorTrain[1,],6),signif(validationCurveList$errorVal[1,],6))
for(lambdaIndex in 2:dim(validationCurveList$lambdaVec)[1]) {
  dispMat = rbind(dispMat,c(validationCurveList$lambdaVec[lambdaIndex,],signif(validationCurveList$errorTrain[lambdaIndex,],6),signif(validationCurveList$errorVal[lambdaIndex,],6)))
}
colnames(dispMat) = c("lambda","Train Error","Validation Error")
dispMat = as.table(dispMat)
dispMat
returnCode <- readKey()
