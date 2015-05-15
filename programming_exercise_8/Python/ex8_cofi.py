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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import fmin_ncg

# Current iteration of fmin_ncg
Nfeval = 1

class InsufficientFeatures(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class InsufficientTrainingExamples(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# Display current iteration of fmin_ncg
def callbackFMinNcg(Xi):
    "Display current iteration of fmin_ncg"
    global Nfeval
    print("Nfeval = %d" % Nfeval)
    Nfeval += 1

# Compute regularized cost function J(\theta)
def computeCost(theta,y,R,numTrainEx,lamb,numUsers,numMovies,numFeatures):
    "Compute regularized cost function J(\theta)"
    totalNumFeatures = numFeatures*(numUsers+numMovies)
    if (totalNumFeatures == 0):
        raise InsufficientFeatures('totalNumFeatures = 0')
    theta = np.reshape(theta,(totalNumFeatures,1),order='F')
    paramsVec = theta[0:(numUsers*numFeatures),:]
    paramsVecSquared = np.power(paramsVec,2)
    featuresVec = theta[(numUsers*numFeatures):totalNumFeatures,:]
    featuresVecSquared = np.power(featuresVec,2)
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    paramsMat = np.reshape(paramsVec,(numUsers,numFeatures),order='F')
    featuresMat = np.reshape(featuresVec,(numMovies,numFeatures),order='F')
    yMat = np.multiply((np.ones((numUsers,numMovies))-np.transpose(R)),(np.dot(paramsMat,np.transpose(featuresMat))))+np.transpose(y)
    costFunctionMat = np.dot(paramsMat,np.transpose(featuresMat))-yMat
    costFunctionVec = costFunctionMat[:,0]
    for columnIndex in range(1,costFunctionMat.shape[1]):
        costFunctionVec = np.vstack((costFunctionVec,costFunctionMat[:,columnIndex]))
    jTheta = (1/2)*np.sum(np.power(costFunctionVec,2))
    jThetaReg = jTheta+(lamb/2)*(np.sum(paramsVecSquared)+np.sum(featuresVecSquared))

    return(jThetaReg)

# Compute gradient of regularized cost function J(\theta)
def computeGradient(theta,y,R,numTrainEx,lamb,numUsers,numMovies,numFeatures):
    "Compute gradient of regularized cost function J(\theta)"
    totalNumFeatures = numFeatures*(numUsers+numMovies)
    if (totalNumFeatures == 0):
        raise InsufficientFeatures('totalNumFeatures = 0')
    theta = np.reshape(theta,(totalNumFeatures,1),order='F')
    paramsVec = theta[0:(numUsers*numFeatures),:]
    paramsVecSquared = np.power(paramsVec,2)
    featuresVec = theta[(numUsers*numFeatures):totalNumFeatures,:]
    featuresVecSquared = np.power(featuresVec,2)
    paramsMat = np.reshape(paramsVec,(numUsers,numFeatures),order='F')
    featuresMat = np.reshape(featuresVec,(numMovies,numFeatures),order='F')
    yMat = np.multiply((np.ones((numUsers,numMovies))-np.transpose(R)),(np.dot(paramsMat,np.transpose(featuresMat))))+np.transpose(y)
    diffMat = np.transpose(np.dot(paramsMat,np.transpose(featuresMat))-yMat)
    gradParamsArray = np.zeros((numUsers*numFeatures,1))
    gradParamsArrayReg = np.zeros((numUsers*numFeatures,1))
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    for gradIndex in range(0,numUsers*numFeatures):
        userIndex = 1+np.mod(gradIndex,numUsers)
        featureIndex = 1+((gradIndex-np.mod(gradIndex,numUsers))/numUsers)
        gradParamsArray[gradIndex] = np.sum(np.multiply(diffMat[:,userIndex-1],featuresMat[:,featureIndex-1]))
        gradParamsArrayReg[gradIndex] = gradParamsArray[gradIndex]+lamb*paramsVec[gradIndex]
    gradFeaturesArray = np.zeros((numMovies*numFeatures,1))
    gradFeaturesArrayReg = np.zeros((numMovies*numFeatures,1))
    for gradIndex in range(0,numMovies*numFeatures):
        movieIndex = 1+np.mod(gradIndex,numMovies)
        featureIndex = 1+((gradIndex-np.mod(gradIndex,numMovies))/numMovies)
        gradFeaturesArray[gradIndex] = np.sum(np.multiply(diffMat[movieIndex-1,:],np.transpose(paramsMat[:,featureIndex-1])))
        gradFeaturesArrayReg[gradIndex] = gradFeaturesArray[gradIndex]+lamb*featuresVec[gradIndex]
    gradArrayReg = np.zeros((numFeatures*(numUsers+numMovies),1))
    gradArrayReg[0:(numUsers*numFeatures),:] = gradParamsArrayReg
    gradArrayReg[(numUsers*numFeatures):((numUsers+numMovies)*numFeatures),:] = gradFeaturesArrayReg

    gradArrayRegFlat = np.ndarray.flatten(gradArrayReg)
    return(gradArrayRegFlat)

# Aggregate computed cost and gradient
def computeCostGradList(y,R,theta,lamb,numUsers,numMovies,numFeatures):
    "Aggregate computed cost and gradient"
    numTrainEx = y.shape[0]
    jThetaReg = computeCost(theta,y,R,numTrainEx,lamb,numUsers,numMovies,numFeatures)
    gradArrayRegFlat = computeGradient(theta,y,R,numTrainEx,lamb,numUsers,numMovies,numFeatures)
    gradArrayReg = np.reshape(gradArrayRegFlat,(numFeatures*(numUsers+numMovies),1),order='F')

    returnList = {'jThetaReg': jThetaReg,'gradArrayReg': gradArrayReg}

    return(returnList)

# Return list of movies
def loadMovieList():
    "Return list of movies"
    movieIdsFile = open('../movie_ids.txt','r')
    movieList = []
    for line in movieIdsFile:
        tokenCount = 0
        movieId = ""
        for token in line.split():
            if (tokenCount > 0):
                movieId += token
                movieId += " "
            tokenCount = tokenCount + 1
        movieId = movieId.lstrip(' ')
        movieId = movieId.rstrip(' ')
        movieList.append(movieId)
    movieIdsFile.close()

    return(movieList)

# Normalize movie ratings
def normalizeRatings(Y,R):
    "Normalize movie ratings"
    numMovies = Y.shape[0]
    numUsers = Y.shape[1]
    YMean = np.zeros((numMovies,1))
    YNorm = np.zeros((numMovies,numUsers))
    for movieIndex in range(0,numMovies):
        ratedUsers = np.where(R[movieIndex,:] == 1)
        YMean[movieIndex,] = np.sum(Y[movieIndex,:])/np.sum(R[movieIndex,:])
        YNorm[movieIndex,ratedUsers] = Y[movieIndex,ratedUsers]-YMean[movieIndex,:]

    returnList = {'YMean': YMean,'YNorm': YNorm}

    return(returnList)
	
# Main function
def main():
    "Main function"
    print("Loading movie ratings dataset.")
    ratingsData = np.genfromtxt("../ratingsMat.txt",delimiter=",")
    numMovies = ratingsData.shape[0]
    numUsers = ratingsData.shape[1]
    ratingsMat = ratingsData[:,0:numUsers]
    indicatorData = np.genfromtxt("../indicatorMat.txt",delimiter=",")
    indicatorMat = indicatorData[:,0:numUsers]
    print("Average rating for movie 1 (Toy Story): %.6f / 5" % np.mean(ratingsMat[0,np.where(indicatorMat[0,:] == 1)]))
    plt.imshow(np.transpose(ratingsMat),cmap=cm.coolwarm,origin='lower')
    plt.ylabel('Movies',fontsize=18)
    plt.xlabel('Users',fontsize=18)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()
    input("Program paused. Press enter to continue.")

    # Compute cost function for a subset of users, movies and features
    featuresData = np.genfromtxt("../featuresMat.txt",delimiter=",")
    numFeatures = featuresData.shape[1]
    featuresMat = featuresData[:,0:numFeatures]
    parametersData = np.genfromtxt("../parametersMat.txt",delimiter=",")
    parametersMat = parametersData[:,0:numFeatures]
    otherParamsData = np.genfromtxt("../otherParams.txt",delimiter=",")
    subsetNumUsers = 4
    subsetNumMovies = 5
    subsetNumFeatures = 3
    subsetFeaturesMat = featuresMat[0:subsetNumMovies,0:subsetNumFeatures]
    subsetParametersMat = parametersMat[0:subsetNumUsers,0:subsetNumFeatures]
    subsetRatingsMat = ratingsMat[0:subsetNumMovies,0:subsetNumUsers]
    subsetIndicatorMat = indicatorMat[0:subsetNumMovies,0:subsetNumUsers]
    parametersVec = subsetParametersMat[:,0]
    for featureIndex in range(1,subsetNumFeatures):
        parametersVec = np.hstack((parametersVec,subsetParametersMat[:,featureIndex]))
    parametersVec = np.reshape(parametersVec,(subsetNumUsers*subsetNumFeatures,1),order='F')
    featuresVec = subsetFeaturesMat[:,0]
    for featureIndex in range(1,subsetNumFeatures):
        featuresVec = np.hstack((featuresVec,subsetFeaturesMat[:,featureIndex]))
    featuresVec = np.reshape(featuresVec,(subsetNumMovies*subsetNumFeatures,1),order='F')
    thetaVec = np.zeros((subsetNumFeatures*(subsetNumUsers+subsetNumMovies),1))
    thetaVec[0:(subsetNumUsers*subsetNumFeatures),:] = parametersVec
    thetaVec[(subsetNumUsers*subsetNumFeatures):((subsetNumUsers+subsetNumMovies)*subsetNumFeatures),:] = featuresVec
    lamb = 0
    initComputeCostList = computeCostGradList(subsetRatingsMat,subsetIndicatorMat,thetaVec,lamb,subsetNumUsers,subsetNumMovies,subsetNumFeatures)
    print("Cost at loaded parameters: %.6f (this value should be about 22.22)" % initComputeCostList['jThetaReg'])
    input("Program paused. Press enter to continue.")

    # Compute regularized cost function for a subset of users, movies and features
    lamb = 1.5
    initComputeCostList = computeCostGradList(subsetRatingsMat,subsetIndicatorMat,thetaVec,lamb,subsetNumUsers,subsetNumMovies,subsetNumFeatures)
    print("Cost at loaded parameters (lambda = 1.5): %.6f (this value should be about 31.34)" % initComputeCostList['jThetaReg'])
    input("Program paused. Press enter to continue.")

    # Add ratings that correspond to a new user
    movieList = loadMovieList()
    myRatings = np.zeros((numMovies,1))
    myRatings[0,:] = 4
    myRatings[97,:] = 2
    myRatings[6,:] = 3
    myRatings[11,:] = 5
    myRatings[53,:] = 4
    myRatings[63,:] = 5
    myRatings[65,:] = 3
    myRatings[68,:] = 5
    myRatings[182,:] = 4
    myRatings[225,:] = 5
    myRatings[354,:] = 5
    print("New user ratings:")
    for movieIndex in range(0,numMovies):
        if (myRatings[movieIndex,:] > 0):
            print("Rated %d for %s" % (myRatings[movieIndex,:],movieList[movieIndex]))
    input("Program paused. Press enter to continue.")

    # Train collaborative filtering model
    print("Training collaborative filtering...")
    ratingsMat = np.c_[myRatings,ratingsMat]
    myIndicators = (myRatings != 0)
    indicatorMat = np.c_[myIndicators,indicatorMat]
    normalizeRatingsList = normalizeRatings(ratingsMat,indicatorMat)
    numUsers = ratingsMat.shape[1]
    np.random.seed(1)
    parametersVec = np.random.normal(size=(numUsers*numFeatures,1))
    featuresVec = np.random.normal(size=(numMovies*numFeatures,1))
    thetaVec = np.zeros((numFeatures*(numUsers+numMovies),1))
    thetaVec[0:(numUsers*numFeatures),:] = parametersVec
    thetaVec[(numUsers*numFeatures):((numUsers+numMovies)*numFeatures),:] = featuresVec
    thetaVecFlat = np.ndarray.flatten(thetaVec)
    lamb = 10
    fminNcgOut = fmin_ncg(computeCost,thetaVecFlat,args=(ratingsMat,indicatorMat,numMovies,lamb,numUsers,numMovies,numFeatures),fprime=computeGradient,callback=callbackFMinNcg,maxiter=100,full_output=1)
    thetaOpt = np.reshape(fminNcgOut[0],(numFeatures*(numUsers+numMovies),1),order='F')
    finalParametersVec = thetaOpt[0:(numUsers*numFeatures),:]
    finalFeaturesVec = thetaOpt[(numUsers*numFeatures):((numUsers+numMovies)*numFeatures),:]
    print("Recommender system learning completed.")
    input("Program paused. Press enter to continue.")

    # Make recommendations
    finalParametersMat = np.reshape(finalParametersVec,(numUsers,numFeatures),order='F')
    finalFeaturesMat = np.reshape(finalFeaturesVec,(numMovies,numFeatures),order='F')
    predVals = np.dot(finalFeaturesMat,np.transpose(finalParametersMat))
    myPredVals = np.reshape(predVals[:,0],(numMovies,1),order='F')
    myPredVals += normalizeRatingsList['YMean']
    sortMyPredVals = np.sort(myPredVals,axis=None)[::-1]
    sortMyPredIndices = np.reshape(np.argsort(myPredVals,axis=None)[::-1],(numMovies,1),order='F')
    print("Top recommendations for you:")
    for topMovieIndex in range(0,10):
        topMovie = sortMyPredIndices[topMovieIndex]
        print("Predicting rating %.1f for movie %s" % (sortMyPredVals[topMovieIndex],movieList[topMovie]))
    print("Original ratings provided:")
    for movieIndex in range(0,numMovies):
        if (myRatings[movieIndex,:] > 0):
            print("Rated %d for %s" % (myRatings[movieIndex,:],movieList[movieIndex]))

# Call main function
if __name__ == "__main__":
    main()
