# Copyright (C) 2015  Caleb Lo
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
# Programming Exercise 8: Recommender Systems
# Problem: Apply collaborative filtering to a dataset of movie ratings

# Load packages
library(pracma)

# Read key press
readKey <- function(){
  cat ("Program paused. Press enter to continue.")
  line <- readline()
  return(0)
}

# Compute regularized cost function J(\theta)
computeCost <- function(theta,y,R,numTrainEx,lambda,numUsers,numMovies,numFeatures){
  theta = as.matrix(theta)
  paramsVec = theta[1:(numUsers*numFeatures),]
  paramsVecSquared = paramsVec^2
  featuresVec = theta[(numUsers*numFeatures+1):dim(theta)[1],]
  featuresVecSquared = featuresVec^2
  if (numTrainEx > 0) {
    paramsMat = matrix(paramsVec,nrow=numUsers,byrow=FALSE)
    featuresMat = matrix(featuresVec,nrow=numMovies,byrow=FALSE)
    yMat = (matrix(1,numUsers,numMovies)-t(R))*(paramsMat%*%t(featuresMat))+t(y)
    jTheta = (1/2)*sum((cbind(c(paramsMat%*%t(featuresMat)-yMat)))^2)
    jThetaReg = jTheta+(lambda/2)*(sum(paramsVecSquared)+sum(featuresVecSquared))
  }
  else
    stop('Insufficient training examples')
  return(jThetaReg)
}

# Compute gradient of regularized cost function J(\theta)
computeGradient <- function(theta,y,R,numTrainEx,lambda,numUsers,numMovies,numFeatures){
  theta = as.matrix(theta)
  if (numFeatures > 0) {
    if (numTrainEx > 0) {
      paramsVec = theta[1:(numUsers*numFeatures),]
      featuresVec = theta[(numUsers*numFeatures+1):dim(theta)[1],]
      paramsMat = matrix(paramsVec,nrow=numUsers,byrow=FALSE)
      featuresMat = matrix(featuresVec,nrow=numMovies,byrow=FALSE)
      yMat = (matrix(1,numUsers,numMovies)-t(R))*(paramsMat%*%t(featuresMat))+t(y)
      diffMat = t(paramsMat%*%t(featuresMat)-yMat)
      gradParamsArray = matrix(0,(numUsers*numFeatures),1)
      gradParamsArrayReg = matrix(0,(numUsers*numFeatures),1)
      for(gradIndex in 1:(numUsers*numFeatures)) {
        userIndex = 1+((gradIndex-1) %% numUsers)
        featureIndex = 1+((gradIndex-1) %/% numUsers)
        gradParamsArray[gradIndex] = sum(diffMat[,userIndex]*featuresMat[,featureIndex])
        gradParamsArrayReg[gradIndex] = gradParamsArray[gradIndex]+lambda*paramsVec[gradIndex]
      }
      gradFeaturesArray = matrix(0,(numMovies*numFeatures),1)
      gradFeaturesArrayReg = matrix(0,(numMovies*numFeatures),1)
      for(gradIndex in 1:(numMovies*numFeatures)) {
        movieIndex = 1+((gradIndex-1) %% numMovies)
        featureIndex = 1+((gradIndex-1) %/% numMovies)
        gradFeaturesArray[gradIndex] = sum(diffMat[movieIndex,]*t(paramsMat[,featureIndex]))
        gradFeaturesArrayReg[gradIndex] = gradFeaturesArray[gradIndex]+lambda*featuresVec[gradIndex]
      }
      gradArrayReg = t(cbind(as.matrix(t(gradParamsArrayReg)),as.matrix(t(gradFeaturesArrayReg))))
    }
    else
      stop('Insufficient training examples')
  }
  else
    stop('Insufficient features')
  return(gradArrayReg)
}

# Aggregate computed cost and gradient
computeCostGradList <- function(y,R,theta,lambda,numUsers,numMovies,numFeatures){
  numTrainEx = dim(y)[1]
  jThetaReg <- computeCost(theta,y,R,numTrainEx,lambda,numUsers,numMovies,numFeatures)
  gradArrayReg <- computeGradient(theta,y,R,numTrainEx,lambda,numUsers,numMovies,numFeatures)
  returnList = list("jThetaReg"=jThetaReg,"gradArrayReg"=gradArrayReg)
  return(returnList)
}

# Return list of movies
loadMovieList <- function(){
  movieIds = readLines("../movie_ids.txt")
  numMovies = length(movieIds)
  movieList = matrix(0,numMovies,1)
  for(movieId in 1:numMovies) {
    movieIdStringSplit = as.vector(strsplit(movieIds[movieId]," "))
    numStrings = length(movieIdStringSplit[[1]])
    movieList[movieId,] = movieIdStringSplit[[1]][2]
    for(stringIndex in 3:numStrings) {
      movieList[movieId,] = paste(movieList[movieId,],movieIdStringSplit[[1]][stringIndex])
      movieList[movieId,] = sub("^\\s+","",movieList[movieId,])
      movieList[movieId,] = sub("\\s+$","",movieList[movieId,])
    }
  }
  return(movieList)
}

# Normalize movie ratings
normalizeRatings <- function(Y,R){
  numMovies = dim(Y)[1]
  numUsers = dim(Y)[2]
  YMean = matrix(0,numMovies,1)
  YNorm = matrix(0,numMovies,numUsers)
  for(movieIndex in 1:numMovies) {
    ratedUsers = which(R[movieIndex,] == 1)
    YMean[movieIndex,] = sum(Y[movieIndex,])/sum(R[movieIndex,])
    YNorm[movieIndex,ratedUsers] = Y[movieIndex,ratedUsers]-YMean[movieIndex,]
  }
  returnList = list("YMean"=YMean,"YNorm"=YNorm)
  return(returnList)
}

# Use setwd() to set working directory to directory that contains this source file
# Load file into R
print(sprintf("Loading movie ratings dataset."))
ratingsData = read.csv("../ratingsMat.txt",header=FALSE)
ratingsMat = as.matrix(ratingsData)
indicatorData = read.csv("../indicatorMat.txt",header=FALSE)
indicatorMat = as.matrix(indicatorData)
print(sprintf("Average rating for movie 1 (Toy Story): %.6f / 5",mean(ratingsMat[1,which(indicatorMat[1,] == 1)])))
image(ratingsMat,col=topo.colors(6),axes=FALSE,xlab="Users",ylab="Movies")
returnCode <- readKey()

# Compute cost function for a subset of users, movies and features
featuresData = read.csv("../featuresMat.txt",header=FALSE)
featuresMat = as.matrix(featuresData)
parametersData = read.csv("../parametersMat.txt",header=FALSE)
parametersMat = as.matrix(parametersData)
otherParamsData = read.csv("../otherParams.txt",header=FALSE)
numUsers = 4
numMovies = 5
numFeatures = 3
subsetFeaturesMat = featuresMat[1:numMovies,1:numFeatures]
subsetParametersMat = parametersMat[1:numUsers,1:numFeatures]
subsetRatingsMat = ratingsMat[1:numMovies,1:numUsers]
subsetIndicatorMat = indicatorMat[1:numMovies,1:numUsers]
parametersVec = cbind(c(subsetParametersMat))
featuresVec = cbind(c(subsetFeaturesMat))
thetaVec = t(cbind(as.matrix(t(parametersVec)),as.matrix(t(featuresVec))))
lambda = 0
initComputeCostList <- computeCostGradList(subsetRatingsMat,subsetIndicatorMat,thetaVec,lambda,numUsers,numMovies,numFeatures)
print(sprintf("Cost at loaded parameters: %.6f (this value should be about 22.22)",initComputeCostList$jTheta))
returnCode <- readKey()

# Compute regularized cost function for a subset of users, movies and features
lambda = 1.5
initComputeCostList <- computeCostGradList(subsetRatingsMat,subsetIndicatorMat,thetaVec,lambda,numUsers,numMovies,numFeatures)
print(sprintf("Cost at loaded parameters (lambda = 1.5): %.6f (this value should be about 31.34)",initComputeCostList$jTheta))
returnCode <- readKey()

# Add ratings that correspond to a new user
movieList <- loadMovieList()
numMovies = dim(movieList)[1]
myRatings = matrix(0,numMovies,1)
myRatings[1,] = 4
myRatings[98,] = 2
myRatings[7,] = 3
myRatings[12,] = 5
myRatings[54,] = 4
myRatings[64,] = 5
myRatings[66,] = 3
myRatings[69,] = 5
myRatings[183,] = 4
myRatings[226,] = 5
myRatings[355,] = 5
print(sprintf("New user ratings:"))
for(movieIndex in 1:numMovies) {
  if (myRatings[movieIndex,] > 0) {
    print(sprintf("Rated %d for %s",myRatings[movieIndex,],movieList[movieIndex,]))
  }
}
returnCode <- readKey()

# Train collaborative filtering model
print(sprintf("Training collaborative filtering..."))
ratingsMat = cbind(myRatings,ratingsMat)
myIndicators = as.matrix(as.numeric((myRatings != 0)),numMovies,1)
indicatorMat = cbind(myIndicators,indicatorMat)
normalizeRatingsList <- normalizeRatings(ratingsMat,indicatorMat)
numUsers = dim(indicatorMat)[2]
numFeatures = 10
parametersVec = as.matrix(rnorm(numUsers*numFeatures))
featuresVec = as.matrix(rnorm(numMovies*numFeatures))
thetaVec = as.matrix(t(cbind(as.matrix(t(parametersVec)),as.matrix(t(featuresVec)))))
lambda = 10
numTrainEx = numMovies
optimResult <- optim(thetaVec,fn=computeCost,gr=computeGradient,ratingsMat,indicatorMat,numTrainEx,lambda,numUsers,numMovies,numFeatures,method="BFGS",control=list(maxit=1600,trace=TRUE,REPORT=1))
finalParametersVec = as.matrix(optimResult$par[1:(numUsers*numFeatures),])
finalFeaturesVec = as.matrix(optimResult$par[(numUsers*numFeatures+1):dim(thetaVec)[1],])
print(sprintf("Recommender system learning completed."))
returnCode <- readKey()

# Make recommendations
finalParametersMat = matrix(finalParametersVec,nrow=numUsers)
finalFeaturesMat = matrix(finalFeaturesVec,nrow=numMovies)
predVals = finalFeaturesMat%*%t(finalParametersMat)
myPredVals = predVals[,1]+normalizeRatingsList$YMean
sortMyPredVals = sort.int(myPredVals,decreasing=TRUE,index.return=TRUE)
print(sprintf("Top recommendations for you:"))
for(topMovieIndex in 1:10) {
  topMovie = sortMyPredVals$ix[topMovieIndex]
  print(sprintf("Predicting rating %.1f for movie %s",sortMyPredVals$x[topMovieIndex],movieList[topMovie,]))
}
print(sprintf("Original ratings provided:"))
for(movieIndex in 1:numMovies) {
  if (myRatings[movieIndex,] > 0) {
    print(sprintf("Rated %d for %s",myRatings[movieIndex,],movieList[movieIndex,]))
  }
}
