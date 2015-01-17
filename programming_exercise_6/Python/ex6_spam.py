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
# Programming Exercise 6: Spam Classification
# Problem: Use SVMs to determine whether various e-mails are spam (or non-spam)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
import Stemmer
import string
from sklearn import svm

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

# Return array of words in a vocabulary list
def getVocabList():
    "Return array of words in a vocabulary list"
    vocabFile = open('../vocab.txt','r')
    vocabList = []
    tokenCount = 0
    for line in vocabFile:
        for token in line.split():
            if (np.mod(tokenCount,2) == 1):
                vocabList.append(token)
            tokenCount = tokenCount + 1
    vocabFile.close()

    return(vocabList)

# Preprocess e-mail and return list of word indices
def processEmail(fileContents):
    "Preprocess e-mail and return list of word indices"
    vocabList = getVocabList()
    lowerFilCont = []
    for listIndex in range(0,len(fileContents)):
        lowerFilCont.append(fileContents[listIndex].lower())
        if (lowerFilCont[listIndex] == '\n'):
            lowerFilCont[listIndex] = ' '
    lowerFilContStr = lowerFilCont[0]
    for listIndex in range(0,len(lowerFilCont)):
        lowerFilContStr = lowerFilContStr + lowerFilCont[listIndex]
    modFileContents = re.sub('<[^<>]+>',' ',lowerFilContStr)
    modFileContents = re.sub('[0-9]+','number',modFileContents)
    modFileContents = re.sub('(http|https)://[^ ]+','httpaddr',modFileContents)
    modFileContents = re.sub('[^ ]+@[^ ]+','emailaddr',modFileContents)
    modFileContents = re.sub('[$]+','dollar',modFileContents)
    print("==== Processed Email ====")

    # Remove punctuation in processed e-mail
    exclude = set(string.punctuation)
    modFileContents = ''.join(ch for ch in modFileContents if ch not in exclude)
    wordsModFileContents = modFileContents.split()
    wordIndices = []

    # Apply porterStemmer in PyStemmer package
    stemmer = Stemmer.Stemmer('porter')
    for wordIndex in range(0,len(wordsModFileContents)):
        currWord = re.sub('[^a-zA-Z0-9]','',wordsModFileContents[wordIndex])
        stemCurrWord = stemmer.stemWord(currWord)
        if (len(stemCurrWord) >= 1) :

            # Search through vocabList for stemmed word
            for vocabListIndex in range(0,len(vocabList)):
                if (stemCurrWord == vocabList[vocabListIndex]):
                    wordIndices.append(vocabListIndex)

            # Display stemmed word
            print("%s" % stemCurrWord)
    print("=========================")

    return(wordIndices)

# Process list of word indices and return feature vector
def emailFeatures(emailWordIndices):
    "Process list of word indices and return feature vector"
    numDictWords = 1899
    featuresVec = np.zeros((numDictWords,1))
    for wordIndex in range(0,len(emailWordIndices)):
        featuresVec[emailWordIndices[wordIndex]] = 1

    return(featuresVec)

# Main function
def main():
    "Main function"
    print("Preprocessing sample email (emailSample1.txt)")
    emailSample1 = open('../emailSample1.txt','r')
    fileContents = []
    while True:
        c = emailSample1.read(1)
        if not c:
            break
        else:
            fileContents.append(c)
    emailSample1.close()

    # Extract features from file
    emailWordIndices = processEmail(fileContents)
    print("Word Indices: ")
    for wordIndex in range(0,len(emailWordIndices)):
        print("%d" % emailWordIndices[wordIndex])
    input("Program paused. Press enter to continue.")
    print("")
    print("Extracting features from sample email (emailSample1.txt)")
    emailSample1 = open('../emailSample1.txt','r')
    fileContents = []
    while True:
        c = emailSample1.read(1)
        if not c:
            break
        else:
            fileContents.append(c)
    emailSample1.close()
    emailWordIndices = processEmail(fileContents)
    featuresVec = emailFeatures(emailWordIndices)
    print("Length of feature vector: %d" % featuresVec.shape[0])
    print("Number of non-zero entries: %d" % np.sum(featuresVec > 0))
    input("Program paused. Press enter to continue.")
    print("")

    # Train a linear SVM for spam classification
    spamTrainData = np.genfromtxt("../spamTrain.txt",delimiter=",")
    numTrainEx = spamTrainData.shape[0]
    numFeatures = spamTrainData.shape[1]-1
    xMat = spamTrainData[:,0:numFeatures]
    yVec = spamTrainData[:,numFeatures]
    print("Training Linear SVM (Spam Classification)")
    print("(this may take 1 to 2 minutes) ...")
    svmModel = svm.SVC(C=0.1,kernel='linear')
    svmModel.fit(xMat,yVec)
    predVec = svmModel.predict(xMat)
    numPredMatch = 0
    for exIndex in range(0,numTrainEx):
        if (predVec[exIndex] == yVec[exIndex]):
            numPredMatch = numPredMatch + 1
    print("Training Accuracy: %.6f" % (100*numPredMatch/numTrainEx))

    # Test this linear spam classifier
    spamTestData = np.genfromtxt("../spamTest.txt",delimiter=",")
    numTestEx = spamTestData.shape[0]
    xTestMat = spamTestData[:,0:numFeatures]
    yTestVec = spamTestData[:,numFeatures]
    print("Evaluating the trained Linear SVM on a test set ...")
    predVec = svmModel.predict(xTestMat)
    numPredMatch = 0
    for exIndex in range(0,numTestEx):
        if (predVec[exIndex] == yTestVec[exIndex]):
            numPredMatch = numPredMatch + 1
    print("Test Accuracy: %.6f" % (100*numPredMatch/numTestEx))
    input("Program paused. Press enter to continue.")
    print("")

    # Determine the top predictors of spam
    svmModelWeightsSorted = np.sort(svmModel.coef_,axis=None)[::-1]
    svmModelWeightsSortedIndices = np.argsort(svmModel.coef_,axis=None)[::-1]
    vocabList = getVocabList()
    print("Top predictors of spam:")
    for predIndex in range(0,15):
        print("%s (%.6f)" % (vocabList[svmModelWeightsSortedIndices[predIndex]],svmModelWeightsSorted[predIndex]))
    input("Program paused. Press enter to continue.")
    print("")

    # Test this linear spam classifier on another e-mail
    spamSample1 = open('../spamSample1.txt','r')
    fileContents = []
    while True:
        c = spamSample1.read(1)
        if not c:
            break
        else:
            fileContents.append(c)
    spamSample1.close()
    emailWordIndices = processEmail(fileContents)
    featuresVec = emailFeatures(emailWordIndices)
    predVec = svmModel.predict(np.transpose(featuresVec))
    print("Processed spamSample1.txt")
    print("Spam Classification: %d" % predVec[0])
    print("(1 indicates spam, 0 indicates not spam)")

# Call main function
if __name__ == "__main__":
    main()
