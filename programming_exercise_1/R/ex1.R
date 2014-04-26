# Machine Learning
# Programming Exercise 1: Linear Regression
# Problem: Predict profits for a food truck given data for profits/populations of various cities
# Linear regression with one variable

# Load packages
library(pracma)

# Plot data
plotData <- function(x,y){
  plot(x,y,col="red",pch="x",xlab="Population of City in 10,000s",ylab="Profit in $10,000s")
  return(0)
}

# Read key press
readKey <- function(){
  cat ("Program paused.  Press enter to continue.")
  line <- readline()
  return(0)
}

# Compute cost function J(\theta)
computeCost <- function(X,y,theta){
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
      stop('Insufficient iterations')
  }
  else
    stop('Insufficient training examples')
  return(theta)
}

# Use setwd() to set working directory to directory that contains this source file
# Load file into R
foodTruckData = read.csv("../foodTruckData.txt",header=FALSE)

# Plot data
print("Plotting data ...")
returnCode <- plotData(foodTruckData[,"V1"],foodTruckData[,"V2"])
returnCode <- readKey()
numTrainEx = dim(foodTruckData)[1]
onesVec = t(t(rep(1,numTrainEx)))
xMat = cbind(onesVec,foodTruckData[,"V1"])
thetaVec = t(t(rep(0,2)))
yVec = cbind(foodTruckData[,"V2"])

# Compute initial cost
print("Running Gradient Descent ...")
initCost <- computeCost(xMat,yVec,thetaVec)
print(sprintf("ans = %.3f",initCost))

# Run gradient descent
iterations = 1500
alpha = 0.01
thetaFinal <- gradientDescent(xMat,yVec,thetaVec,alpha,iterations)
print(sprintf("Theta found by gradient descent: %s",paste(format(round(thetaFinal,6),nsmall=6),collapse=" ")))
lines(foodTruckData[,"V1"],xMat%*%thetaFinal,col="blue",pch="-")
plotLegend <- legend('bottomright',col=c("red","blue"),pch=c("x","-"),legend=c("",""),bty="n",trace=TRUE)
text(plotLegend$text$x-1,plotLegend$text$y,c('Training data','Linear regression'),pos=2)

# Predict profit for population size of 35000
predProfit1 = matrix(c(1,3.5),nrow=1,ncol=2) %*% thetaFinal
predProfit1Scaled = 10000*predProfit1
print(sprintf("For population = 35,000, we predict a profit of %.6f",predProfit1Scaled))

# Predict profit for population size of 70000
predProfit2 = matrix(c(1,7),nrow=1,ncol=2) %*% thetaFinal
predProfit2Scaled = 10000*predProfit2
print(sprintf("For population = 70,000, we predict a profit of %.6f",predProfit2Scaled))
returnCode <- readKey()

# Surface plot
print("Visualizing J(theta_0, theta_1) ...")
theta0Vals = seq(-10,10,20/99)
theta1Vals = seq(-1,4,5/99)
J_vals = mat.or.vec(dim(cbind(theta0Vals))[1],dim(cbind(theta1Vals))[1])
for(index0 in 1:dim(cbind(theta0Vals))[1]) {
  for(index1 in 1:dim(cbind(theta1Vals))[1]){
    t = rbind(theta0Vals[index0],theta1Vals[index1])
    J_vals[index0,index1] <- computeCost(xMat,yVec,t)
  }
}
persp(theta0Vals,theta1Vals,J_vals,xlab=expression(theta[0]),ylab=expression(theta[1]))

# Contour plot
contour(x=theta0Vals,y=theta1Vals,J_vals,levels=logspace(-2,3,20),drawlabels=FALSE,plot.title = title(xlab=expression(theta[0]),ylab=expression(theta[1])))
points(x=thetaFinal[1,1],y=thetaFinal[2,1],col="red",cex=1.75,pch="x")
