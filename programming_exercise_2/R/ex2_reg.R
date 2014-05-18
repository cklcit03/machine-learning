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
# Problem: Predict chances of acceptance for a microchip given data for 
# acceptance decisions and test scores of various microchips

# Plot data
plotData <- function(X,y){
  positiveIndices = which(y == 1)
  negativeIndices = which(y == 0)
  positiveExamples = cbind(X[positiveIndices,])
  negativeExamples = cbind(X[negativeIndices,])
  plot(positiveExamples[,1],positiveExamples[,2],cex=1.75,pch="+",xlab="",ylab="",xlim=c(-1,1.5),ylim=c(-1,1.5))
  points(x=negativeExamples[,1],y=negativeExamples[,2],col="yellow",bg="yellow",cex=1.75,pch=22,xlab="",ylab="")
  return(0)
}

# Plot decision boundary
plotDecisionBoundary <- function(X,y,theta){
  returnCode <- plotData(cbind(X[,1],X[,2]),y)
  u = seq(-1,1.5,length=50)
  v = seq(-1,1.5,length=50)
  z = mat.or.vec(dim(cbind(u))[1],dim(cbind(v))[1])
  for(index0 in 1:dim(cbind(u))[1]) {
    for(index1 in 1:dim(cbind(v))[1]) {
      z[index0,index1] <- mapFeature(u[index0],v[index1])%*%theta;
    }
  }
  contour(x=u,y=v,z,nlevels=1,zlim=range(0),col="blue",lwd=2,drawlabels=FALSE,add=TRUE)
  return(0)
}

# Read key press
readKey <- function(){
  cat ("Program paused. Press enter to continue.")
  line <- readline()
  return(0)
}

# Add polynomial features to training data
mapFeature <- function(X1,X2){
  degree = 6
  numTrainEx = dim(cbind(X1,X2))[1]
  augXMat = matrix(1,numTrainEx,1)
  for(degIndex1 in 1:degree) {
    for(degIndex2 in 0:degIndex1) {
      augXMat = cbind(augXMat,(X1^(degIndex1-degIndex2))*(X2^degIndex2))
    }
  }
  return(augXMat)
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
  jTheta = (colSums(-y*log(hTheta)-(1-y)*log(1-hTheta)))/numTrainEx
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
    for(gradIndex in 1:numFeatures) {
      gradTermArray[,gradIndex] = (hTheta-y)*X[,gradIndex]
      gradArray[gradIndex] = (sum(gradTermArray[,gradIndex]))/(numTrainEx)
      gradArrayReg[gradIndex] = gradArray[gradIndex]+(lambda/numTrainEx)*theta[gradIndex]
    }
  }
  else
    stop('Insufficient features')
  gradArrayReg[1] = gradArrayReg[1]-(lambda/numTrainEx)*theta[1]
  return(gradArrayReg)
}

# Aggregate computed cost and gradient
computeCostGradList <- function(X,y,theta,lambda){
  numTrainEx = dim(y)[1]
  if (numTrainEx > 0) {
    jThetaReg <- computeCost(theta,X,y,numTrainEx,lambda)
    gradArrayReg <- computeGradient(theta,X,y,numTrainEx,lambda)
  }
  else
    stop('Insufficient training examples')
  returnList = list("jThetaReg"=jThetaReg,"gradArrayReg"=gradArrayReg)
  return(returnList)
}

# Perform label prediction on training data
labelPrediction <- function(X,theta){
  sigmoidArr <- computeSigmoid(X%*%theta)
  p = (sigmoidArr >= 0.5)
  return(p)
}

# Use setwd() to set working directory to directory that contains this source file
# Load file into R
microChipData = read.csv("../microChipData.txt",header=FALSE)

# Plot data
returnCode <- plotData(cbind(microChipData[,"V1"],microChipData[,"V2"]),microChipData[,"V3"])
title(xlab="Microchip Test 1",ylab="Microchip Test 2")
plotLegend <- legend('bottomright',col=c("black","yellow"),pt.bg=c("black","yellow"),pch=c(43,22),pt.cex=1.75,legend=c("",""),bty="n",trace=TRUE)
text(plotLegend$text$x-0.1,plotLegend$text$y,c('y = 1','y = 0'),pos=2)
numTrainEx = dim(microChipData)[1]
numFeatures = dim(microChipData)[2]-1
xMat = cbind(microChipData[,"V1"],microChipData[,"V2"])
yVec = cbind(microChipData[,"V3"])

# Add polynomial features to training data
featureXMat <- mapFeature(xMat[,1],xMat[,2])
thetaVec = t(t(rep(0,dim(featureXMat)[2])))

# Compute initial cost and gradient
lambda = 1
initComputeCostList <- computeCostGradList(featureXMat,yVec,thetaVec,lambda)
print(sprintf("Cost at initial theta (zeros): %.6f",initComputeCostList$jTheta))
returnCode <- readKey()

# Use optim to solve for optimum theta and cost
optimResult <- optim(thetaVec,fn=computeCost,gr=computeGradient,featureXMat,yVec,numTrainEx,lambda,method="BFGS",control=list(maxit=400))

# Plot decision boundary
returnCode <- plotDecisionBoundary(xMat,yVec,optimResult$par)
title(main=sprintf("lambda = %g",lambda),xlab="Microchip Test 1",ylab="Microchip Test 2")
plotLegend <- legend('bottomright',col=c("black","yellow"),pt.bg=c("black","yellow"),pch=c(43,22),pt.cex=1.75,legend=c("",""),bty="n",trace=TRUE)
text(plotLegend$text$x-0.1,plotLegend$text$y,c('y = 1','y = 0'),pos=2)

# Compute accuracy on training set
trainingPredict <- (labelPrediction(featureXMat,optimResult$par)+0)
print(sprintf("Train Accuracy: %.6f",100*apply((trainingPredict == yVec),2,mean)))
