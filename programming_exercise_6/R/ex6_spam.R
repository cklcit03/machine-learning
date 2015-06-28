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

ReadKey <- function() {
  # Reads key press.
  #
  # Args:
  #   None.
  #
  # Returns:
  #   None.
  cat("Program paused. Press enter to continue.")
  line <- readline()
  return(0)
}

GetVocabList <- function() {
  # Returns array of words in a vocabulary list.
  #
  # Args:
  #   None.
  #
  # Returns:
  #   vocabList: Matrix that contains a vocabulary list.
  fullList <- as.matrix(scan("../vocab.txt", what=character()))
  numWords <- (dim(fullList)[1]) / 2
  if (numWords > 0) {
    vocabList <- matrix(0, numWords, 1)
    for (listIndex in 1:numWords) {
      vocabList[listIndex, ] <- fullList[2 * listIndex]
    }
  } else {
    stop('Insufficient number of words in vocabulary list')
  }
  return(vocabList)
}

ProcessEmail <- function(fileContents) {
  # Preprocesses e-mail and returns list of word indices.
  #
  # Args:
  #   fileContents: Vector of characters from an e-mail in a text file.
  #
  # Returns:
  #   wordIndices: Vector of indices of processed words in a vocabulary list.
  numChars <- dim(fileContents)[1]
  if (numChars > 0) {
    lowerFilCont <- matrix(0, numChars, 1)
    for (listIndex in 1:numChars) {
      lowerFilCont[listIndex, ] <- tolower(fileContents[listIndex, ])
      if (lowerFilCont[listIndex, ] == '\n') {
        lowerFilCont[listIndex, ] <- ' '
      }
    }
    lowerFilContStr <- lowerFilCont[1]
    for (listIndex in 2:numChars) {
      lowerFilContStr <- paste(lowerFilContStr, lowerFilCont[listIndex], 
                               sep='')
    }
    modFileContents <- gsub("<[^<>]+>", " ", lowerFilContStr)
    modFileContents <- gsub("[0-9]+", "number", modFileContents)
    modFileContents <- gsub("(http|https)://[^ ]+", "httpaddr", 
                            modFileContents)
    modFileContents <- gsub("[^ ]+@[^ ]+", "emailaddr", modFileContents)
    modFileContents <- gsub("[$]+", "dollar", modFileContents)
    print(sprintf("==== Processed Email ===="))

    # Remove punctuation in processed e-mail
    modFileContents <- gsub("[[:punct:]]", "", modFileContents)
    wordsModFileContents <- strsplit(modFileContents, " ")
    vocabList <- GetVocabList()
    wordIndices <- list()
    for (wordIndex in 1:length(wordsModFileContents[[1]])) {
      if (wordsModFileContents[[1]][wordIndex] != "") {
        currWord <- gsub("[^a-zA-Z0-9]", "", 
                         wordsModFileContents[[1]][wordIndex])

        # Apply porterStemmer in Rstem package
        stemCurrWord <- wordStem(currWord)
        if (nchar(stemCurrWord) >= 1) {

          # Search through vocabList for stemmed word
          for (vocabListIndex in 1:length(vocabList)) {
            if (stemCurrWord == vocabList[vocabListIndex]) {
              wordIndices <- rbind(wordIndices, vocabListIndex)
            }
          }

          # Display stemmed word
          print(sprintf("%s", stemCurrWord))
        }
      }
    }
    print(sprintf("========================="))
  } else {
    stop('Insufficient number of characters in e-mail')
  }
  return(wordIndices)
}

EmailFeatures <- function(emailWordIndices) {
  # Processes list of word indices and returns feature vector.
  #
  # Args:
  #   emailWordIndices: Vector of indices of processed words (from an e-mail) 
  #                     in a vocabulary list.
  #
  # Returns:
  #   featuresVec: Vector of booleans where the i-th entry indicates whether
  #                the i-th word in the vocabulary list occurs in the e-mail of
  #                interest.
  kNumDictWords <- 1899
  featuresVec <- matrix(0, kNumDictWords, 1)
  numEmailWordIndices <- length(emailWordIndices)
  if (numEmailWordIndices > 0) {
    for (wordIndex in 1:length(emailWordIndices)) {
      featuresVec[as.numeric(emailWordIndices[wordIndex])] <- 1
    }
  } else {
    stop('Insufficient number of processed word indices')
  }
  return(featuresVec)
}

# Use setwd() to set working directory to directory that contains this source 
# file
# Load file into R
print(sprintf("Preprocessing sample email (emailSample1.txt)"))
emailSample1 <- file("../emailSample1.txt", "rb")
fileContents <- readChar(emailSample1, 1)
for (cIndex in 2:file.info("../emailSample1.txt")$size) {
  fileContents <- rbind(fileContents, readChar(emailSample1, 1))
}
close(emailSample1)

# Extract features from file
emailWordIndices <- ProcessEmail(fileContents)
print(sprintf("Word Indices: "))
for (wordIndex in 1:length(emailWordIndices)) {
  print(sprintf("%d", as.numeric(emailWordIndices[wordIndex])))
}
returnCode <- ReadKey()
print(sprintf("Extracting features from sample email (emailSample1.txt)"))
emailSample1 <- file("../emailSample1.txt", "rb")
fileContents <- readChar(emailSample1, 1)
for (cIndex in 2:file.info("../emailSample1.txt")$size) {
  fileContents <- rbind(fileContents, readChar(emailSample1, 1))
}
close(emailSample1)
emailWordIndices <- ProcessEmail(fileContents)
featuresVec <- EmailFeatures(emailWordIndices)
print(sprintf("Length of feature vector: %d", length(featuresVec)))
print(sprintf("Number of non-zero entries: %d", sum(featuresVec > 0)))
returnCode <- ReadKey()

# Train a linear SVM for spam classification
spamTrain <- read.csv("../spamTrain.txt", header=FALSE)
xMat <- as.matrix(subset(spamTrain, select=-c(V1900)))
yVec <- as.vector(subset(spamTrain, select=c(V1900)))
print(sprintf("Training Linear SVM (Spam Classification)"))
print(sprintf("(this may take 1 to 2 minutes) ..."))
svmModel <- svm(xMat, y=yVec, scale=FALSE, type="C-classification", 
                kernel="linear", cost=0.1)
svmPred <- predict(svmModel, xMat)
print(sprintf("Training Accuracy: %.6f", 
              100 * apply(((as.numeric(svmPred) - 1) == yVec), 2, mean)))

# Test this linear spam classifier
spamTest <- read.csv("../spamTest.txt", header=FALSE)
xTestMat <- as.matrix(subset(spamTest, select=-c(V1900)))
yTestVec <- as.vector(subset(spamTest, select=c(V1900)))
print(sprintf("Evaluating the trained Linear SVM on a test set ..."))
testPred <- predict(svmModel, xTestMat)
print(sprintf("Test Accuracy: %.6f", 
              100 * apply(((as.numeric(testPred) - 1) == yTestVec), 2, mean)))
returnCode <- ReadKey()

# Determine the top predictors of spam
svmModelWeights <- t(svmModel$coefs) %*% (svmModel$SV)
svmModelWeightsSorted <- sort(svmModelWeights, decreasing=TRUE, 
                              index.return=TRUE)
vocabList <- GetVocabList()
print(sprintf("Top predictors of spam:"))
for (predIndex in 1:15) {
  print(sprintf("%s (%.6f)", vocabList[svmModelWeightsSorted$ix[predIndex]], 
                svmModelWeightsSorted$x[predIndex]))
}
returnCode <- ReadKey()

# Test this linear spam classifier on another e-mail
spamSample1 <- file("../spamSample1.txt", "rb")
fileContents <- readChar(spamSample1, 1)
for (cIndex in 2:file.info("../spamSample1.txt")$size) {
  fileContents <- rbind(fileContents, readChar(spamSample1, 1))
}
close(spamSample1)
emailWordIndices <- ProcessEmail(fileContents)
featuresVec <- EmailFeatures(emailWordIndices)
svmPred <- predict(svmModel, t(featuresVec))
print(sprintf("Processed spamSample1.txt"))
print(sprintf("Spam Classification: %d", as.numeric(svmPred) - 1))
print(sprintf("(1 indicates spam, 0 indicates not spam)"))
