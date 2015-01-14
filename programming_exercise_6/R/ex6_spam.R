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
# Programming Exercise 6: Spam Classification
# Problem: Use SVMs to determine whether various e-mails are spam (or non-spam)

# Load packages
library(e1071)
library(Rstem)

# Read key press
readKey <- function(){
  cat ("Program paused. Press enter to continue.")
  line <- readline()
  return(0)
}

# Return array of words in a vocabulary list
getVocabList <- function(){
  fullList = as.matrix(scan("../vocab.txt",what=character()))
  vocabList = matrix(0,(dim(fullList)[1]/2),1)
  for(listIndex in 1:(dim(fullList)[1]/2)) {
    vocabList[listIndex,] = fullList[2*listIndex]
  }
  return(vocabList)
}

# Preprocess e-mail and return list of word indices
processEmail <- function(fileContents){
  vocabList <- getVocabList()
  lowerFilCont = matrix(0,dim(fileContents)[1],1)
  for(listIndex in 1:dim(fileContents)[1]) {
    lowerFilCont[listIndex,] = tolower(fileContents[listIndex,])
    if (lowerFilCont[listIndex,] == '\n') {
      lowerFilCont[listIndex,] = ' '
    }
  }
  lowerFilContStr = lowerFilCont[1]
  for(listIndex in 2:dim(lowerFilCont)[1]) {
    lowerFilContStr = paste(lowerFilContStr,lowerFilCont[listIndex],sep='')
  }
  modFileContents = gsub("<[^<>]+>"," ",lowerFilContStr)
  modFileContents = gsub("[0-9]+","number",modFileContents)
  modFileContents = gsub("(http|https)://[^ ]+","httpaddr",modFileContents)
  modFileContents = gsub("[^ ]+@[^ ]+","emailaddr",modFileContents)
  modFileContents = gsub("[$]+","dollar",modFileContents)
  print(sprintf("==== Processed Email ===="))
  
  # Remove punctuation in processed e-mail
  modFileContents=gsub("[[:punct:]]","",modFileContents)
  wordsModFileContents=strsplit(modFileContents," ")
  wordIndices = list()
  for(wordIndex in 1:length(wordsModFileContents[[1]])) {
    if (wordsModFileContents[[1]][wordIndex] != "") {
      currWord = gsub("[^a-zA-Z0-9]","",wordsModFileContents[[1]][wordIndex])
      
      # Apply porterStemmer in Rstem package
      stemCurrWord = wordStem(currWord)
      if (nchar(stemCurrWord) >= 1) {
        
        # Search through vocabList for stemmed word
        for(vocabListIndex in 1:length(vocabList)) {
          if (stemCurrWord == vocabList[vocabListIndex]) {
            wordIndices = rbind(wordIndices,vocabListIndex)
          }
        }
        
        # Display stemmed word
        print(sprintf("%s",stemCurrWord))
      }
    }
  }
  print(sprintf("========================="))
  
  return(wordIndices)
}

# Process list of word indices and return feature vector
emailFeatures <- function(emailWordIndices){
  numDictWords = 1899
  featuresVec = matrix(0,numDictWords,1)
  for(wordIndex in 1:length(emailWordIndices)){
    featuresVec[as.numeric(emailWordIndices[wordIndex])] = 1
  }
  
  return(featuresVec)
}

# Use setwd() to set working directory to directory that contains this source file
# Load file into R
print(sprintf("Preprocessing sample email (emailSample1.txt)"))
emailSample1 <- file("../emailSample1.txt","rb")
fileContents = readChar(emailSample1,1)
for(cIndex in 2:file.info("../emailSample1.txt")$size) {
  fileContents = rbind(fileContents,readChar(emailSample1,1))
}
close(emailSample1)

# Extract features from file
emailWordIndices <- processEmail(fileContents)
print(sprintf("Word Indices: "))
for(wordIndex in 1:length(emailWordIndices)) {
  print(sprintf("%d",as.numeric(emailWordIndices[wordIndex])))
}
returnCode <- readKey()
print(sprintf("Extracting features from sample email (emailSample1.txt)"))
emailSample1 <- file("../emailSample1.txt","rb")
fileContents = readChar(emailSample1,1)
for(cIndex in 2:file.info("../emailSample1.txt")$size) {
  fileContents = rbind(fileContents,readChar(emailSample1,1))
}
close(emailSample1)
emailWordIndices <- processEmail(fileContents)
featuresVec <- emailFeatures(emailWordIndices)
print(sprintf("Length of feature vector: %d",length(featuresVec)))
print(sprintf("Number of non-zero entries: %d",sum(featuresVec > 0)))
returnCode <- readKey()

# Train a linear SVM for spam classification
spamTrain = read.csv("../spamTrain.txt",header=FALSE)
xMat = as.matrix(subset(spamTrain,select=-c(V1900)))
yVec = as.vector(subset(spamTrain,select=c(V1900)))
print(sprintf("Training Linear SVM (Spam Classification)"))
print(sprintf("(this may take 1 to 2 minutes) ..."))
svmModel <- svm(xMat,y=yVec,scale=FALSE,type="C-classification",kernel="linear",cost=0.1)
svmPred <- predict(svmModel,xMat)
print(sprintf("Training Accuracy: %.6f",100*apply(((as.numeric(svmPred)-1) == yVec),2,mean)))

# Test this linear spam classifier
spamTest = read.csv("../spamTest.txt",header=FALSE)
xTestMat = as.matrix(subset(spamTest,select=-c(V1900)))
yTestVec = as.vector(subset(spamTest,select=c(V1900)))
print(sprintf("Evaluating the trained Linear SVM on a test set ..."))
testPred <- predict(svmModel,xTestMat)
print(sprintf("Test Accuracy: %.6f",100*apply(((as.numeric(testPred)-1) == yTestVec),2,mean)))
returnCode <- readKey()

# Determine the top predictors of spam
svmModelWeights = t(svmModel$coefs)%*%(svmModel$SV)
svmModelWeightsSorted=sort(svmModelWeights,decreasing=TRUE,index.return=TRUE)
vocabList <- getVocabList()
print(sprintf("Top predictors of spam:"))
for(predIndex in 1:15) {
  print(sprintf("%s (%.6f)",vocabList[svmModelWeightsSorted$ix[predIndex]],svmModelWeightsSorted$x[predIndex]))
}
returnCode <- readKey()

# Test this linear spam classifier on another e-mail
spamSample1 <- file("../spamSample1.txt","rb")
fileContents = readChar(spamSample1,1)
for(cIndex in 2:file.info("../spamSample1.txt")$size) {
  fileContents = rbind(fileContents,readChar(spamSample1,1))
}
close(spamSample1)
emailWordIndices <- processEmail(fileContents)
featuresVec <- emailFeatures(emailWordIndices)
svmPred <- predict(svmModel,t(featuresVec))
print(sprintf("Processed spamSample1.txt"))
print(sprintf("Spam Classification: %d",as.numeric(svmPred)-1))
print(sprintf("(1 indicates spam, 0 indicates not spam)"))
