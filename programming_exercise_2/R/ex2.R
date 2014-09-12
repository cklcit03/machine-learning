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
# Problem: Predict chances of university admission for an applicant given data for 
# admissions decisions and test scores of various applicants

# Plot data
plotData <- function(X,y){
  positiveIndices = which(y == 1)
  negativeIndices = which(y == 0)
  positiveExamples = cbind(X[positiveIndices,])
  negativeExamples = cbind(X[negativeIndices,])
  plot(positiveExamples[,1],positiveExamples[,2],cex=1.75,pch="+",xlab="",ylab="",xlim=c(30,100),ylim=c(30,100))
  points(x=negativeExamples[,1],y=negativeExamples[,2],col="yellow",bg="yellow",cex=1.75,pch=22,xlab="",ylab="")
  return(0)
}

# Plot decision boundary
plotDecisionBoundary <- function(X,y,theta){
  returnCode <- plotData(cbind(X[,2],X[,3]),y)
  yLineVals = (theta[1]+theta[2]*X[,2])/(-1*theta[3])
  lines(cbind(X[,2]),yLineVals,col="blue",pch="-")
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

# Compute cost function J(\theta)
computeCost <- function(theta,X,y,numTrainEx){
  hTheta <- computeSigmoid(X%*%theta)
  if (numTrainEx > 0)
    jTheta = (colSums(-y*log(hTheta)-(1-y)*log(1-hTheta)))/numTrainEx
  else
    stop('Insufficient training examples')
  return(jTheta)  
}

# Compute gradient of cost function J(\theta)
computeGradient <- function(theta,X,y,numTrainEx){
  numFeatures = dim(X)[2]
  hTheta <- computeSigmoid(X%*%theta)
  if (numFeatures > 0) {
    if (numTrainEx > 0) {
      gradArray = matrix(0,numFeatures,1)
      gradTermArray = matrix(0,numTrainEx,numFeatures)
      for(gradIndex in 1:numFeatures) {
        gradTermArray[,gradIndex] = (hTheta-y)*X[,gradIndex]
        gradArray[gradIndex] = (sum(gradTermArray[,gradIndex]))/(numTrainEx)
      }
    }
    else
      stop('Insufficient training examples')
  }
  else
    stop('Insufficient features')
  return(gradArray)
}

# Aggregate computed cost and gradient
computeCostGradList <- function(X,y,theta){
  numTrainEx = dim(y)[1]
  jTheta <- computeCost(theta,X,y,numTrainEx)
  gradArray <- computeGradient(theta,X,y,numTrainEx)
  returnList = list("jTheta"=jTheta,"gradArray"=gradArray)
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
applicantData = read.csv("../applicantData.txt",header=FALSE)

# Plot data
print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")
returnCode <- plotData(cbind(applicantData[,"V1"],applicantData[,"V2"]),applicantData[,"V3"])
title(xlab="Exam 1 score",ylab="Exam 2 score")
plotLegend <- legend('bottomright',col=c("black","yellow"),pt.bg=c("black","yellow"),pch=c(43,22),pt.cex=1.75,legend=c("",""),bty="n",trace=TRUE)
text(plotLegend$text$x-3,plotLegend$text$y,c('Admitted','Not admitted'),pos=2)
returnCode <- readKey()
numTrainEx = dim(applicantData)[1]
numFeatures = dim(applicantData)[2]-1
onesVec = t(t(rep(1,numTrainEx)))
xMat = cbind(onesVec,applicantData[,"V1"],applicantData[,"V2"])
yVec = cbind(applicantData[,"V3"])
thetaVec = t(t(rep(0,numFeatures+1)))

# Compute initial cost and gradient
initComputeCostList <- computeCostGradList(xMat,yVec,thetaVec)
print(sprintf("Cost at initial theta (zeros): %.6f",initComputeCostList$jTheta))
print(sprintf("Gradient at initial theta (zeros): "))
cat(format(round(initComputeCostList$gradArray,6),nsmall=6),sep="\n")
returnCode <- readKey()

# Use optim to solve for optimum theta and cost
optimResult <- optim(thetaVec,fn=computeCost,gr=computeGradient,xMat,yVec,numTrainEx,method="BFGS",control=list(maxit=400))
print(sprintf("Cost at theta found by optim: %.6f",optimResult$value))
print(sprintf("theta: "))
cat(format(round(optimResult$par,6),nsmall=6),sep="\n")
returnCode <- plotDecisionBoundary(xMat,yVec,optimResult$par)
title(xlab="Exam 1 score",ylab="Exam 2 score")
plotLegend <- legend('bottomright',col=c("black","yellow","blue"),pt.bg=c("black","yellow","blue"),pch=c(43,22,45),pt.cex=1.75,legend=c("","",""),bty="n",trace=TRUE)
text(plotLegend$text$x-3,plotLegend$text$y,c('Admitted','Not admitted','Decision Boundary'),pos=2)
returnCode <- readKey()

# Predict admission probability for a student with score 45 on exam 1 and score 85 on exam 2
admissionProb <- computeSigmoid(cbind(1,45,85)%*%optimResult$par)
print(sprintf("For a student with scores 45 and 85, we predict an admission probability of %.6f",admissionProb))

# Compute accuracy on training set
trainingPredict <- (labelPrediction(xMat,optimResult$par)+0)
print(sprintf("Train Accuracy: %.6f",100*apply((trainingPredict == yVec),2,mean)))
returnCode <- readKey()
