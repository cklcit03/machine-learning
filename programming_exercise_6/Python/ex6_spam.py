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
from sklearn import svm
import numpy
import re
import Stemmer
import string


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def get_vocab_list():
    """ Returns array of words in a vocabulary list.

    Args:
      None.

    Returns:
      vocab_list: Matrix that contains a vocabulary list.
    """
    vocab_file = open('../vocab.txt', 'r')
    vocab_list = []
    token_count = 0
    for line in vocab_file:
        for token in line.split():
            if (numpy.mod(token_count, 2) == 1):
                vocab_list.append(token)
            token_count = token_count + 1
    vocab_file.close()
    return vocab_list


def process_email(file_contents):
    """ Preprocesses e-mail and returns list of word indices.

    Args:
      file_contents: Vector of characters from an e-mail in a text file.
   
    Returns:
      word_indices: Vector of indices of processed words in a vocabulary list.

    Raises:
      An error occurs if the number of characters from the e-mail is 0.
      An error occurs if the number of processed words from the vocabulary list
      is 0.
    """
    num_email_chars = len(file_contents)
    if (num_email_chars == 0): raise Error('num_email_chars == 0')
    vocab_list = get_vocab_list()
    num_vocab_list_index = len(vocab_list)
    if (num_vocab_list_index == 0): raise Error('num_vocab_list_index == 0')
    lower_fil_cont = []
    for list_index in range(0, num_email_chars):
        lower_fil_cont.append(file_contents[list_index].lower())
        if (lower_fil_cont[list_index] == '\n'):
            lower_fil_cont[list_index] = ' '
    lower_fil_cont_str = lower_fil_cont[0]
    for list_index in range(0, len(lower_fil_cont)):
        lower_fil_cont_str = lower_fil_cont_str + lower_fil_cont[list_index]
    mod_file_contents = re.sub('<[^<>]+>', ' ', lower_fil_cont_str)
    mod_file_contents = re.sub('[0-9]+', 'number', mod_file_contents)
    mod_file_contents = re.sub('(http|https)://[^ ]+', 'httpaddr',
                               mod_file_contents)
    mod_file_contents = re.sub('[^ ]+@[^ ]+', 'emailaddr', mod_file_contents)
    mod_file_contents = re.sub('[$]+', 'dollar', mod_file_contents)
    print("==== Processed Email ====")

    # Remove punctuation in processed e-mail
    exclude = set(string.punctuation)
    mod_file_contents = (
        ''.join(ch for ch in mod_file_contents if ch not in exclude))
    words_mod_file_contents = mod_file_contents.split()
    word_indices = []

    # Apply porterStemmer in PyStemmer package
    stemmer = Stemmer.Stemmer('porter')
    for word_index in range(0, len(words_mod_file_contents)):
        curr_word = re.sub('[^a-zA-Z0-9]', '',
                           words_mod_file_contents[word_index])
        stem_curr_word = stemmer.stemWord(curr_word)
        if (len(stem_curr_word) >= 1) :

            # Search through vocab_list for stemmed word
            for vocab_list_index in range(0, num_vocab_list_index):
                if (stem_curr_word == vocab_list[vocab_list_index]):
                    word_indices.append(vocab_list_index)

            # Display stemmed word
            print("%s" % stem_curr_word)
    print("=========================")
    return word_indices


def email_features(email_word_indices):
    """ Processes list of word indices and returns feature vector.

    Args:
      email_word_indices: Vector of indices of processed words (from an e-mail) 
                          in a vocabulary list.

    Returns:
      features_vec: Vector of booleans where the i-th entry indicates whether
                    the i-th word in the vocabulary list occurs in the e-mail of
                    interest.

    Raises:
      An error occurs if the number of word indices is 0.
    """
    num_word_indices = len(email_word_indices)
    if (num_word_indices == 0): raise Error('num_word_indices == 0')
    num_dict_words = 1899
    features_vec = numpy.zeros((num_dict_words, 1))
    for word_index in range(0, num_word_indices):
        features_vec[email_word_indices[word_index]] = 1
    return features_vec


def main():
    """ Main function

    Raises:
      An error occurs if the number of processed words from an e-mail is 0.
      An error occurs if the number of training examples is 0.
      An error occurs if the number of test examples is 0.
    """
    print("Preprocessing sample email (emailSample1.txt)")
    email_sample_1 = open('../emailSample1.txt', 'r')
    file_contents = []
    while True:
        c = email_sample_1.read(1)
        if not c:
            break
        else:
            file_contents.append(c)
    email_sample_1.close()

    # Extract features from file
    email_word_indices = process_email(file_contents)
    num_word_indices = len(email_word_indices)
    if (num_word_indices == 0): raise Error('num_word_indices == 0')
    print("Word Indices: ")
    for word_index in range(0, num_word_indices):
        print("%d" % email_word_indices[word_index])
    input("Program paused. Press enter to continue.")
    print("")
    print("Extracting features from sample email (emailSample1.txt)")
    email_sample_1 = open('../emailSample1.txt', 'r')
    file_contents = []
    while True:
        c = email_sample_1.read(1)
        if not c:
            break
        else:
            file_contents.append(c)
    email_sample_1.close()
    email_word_indices = process_email(file_contents)
    features_vec = email_features(email_word_indices)
    print("Length of feature vector: %d" % features_vec.shape[0])
    print("Number of non-zero entries: %d" % numpy.sum(features_vec > 0))
    input("Program paused. Press enter to continue.")
    print("")

    # Train a linear SVM for spam classification
    spam_train_data = numpy.genfromtxt("../spamTrain.txt", delimiter=",")
    num_train_ex = spam_train_data.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex == 0')
    num_features = spam_train_data.shape[1]-1
    x_mat = spam_train_data[:, 0:num_features]
    y_vec = spam_train_data[:, num_features]
    print("Training Linear SVM (Spam Classification)")
    print("(this may take 1 to 2 minutes) ...")
    svm_model = svm.SVC(C=0.1, kernel='linear')
    svm_model.fit(x_mat, y_vec)
    pred_vec = svm_model.predict(x_mat)
    num_pred_match = 0
    for ex_index in range(0, num_train_ex):
        if (pred_vec[ex_index] == y_vec[ex_index]):
            num_pred_match = num_pred_match + 1
    print("Training Accuracy: %.6f" % (100*num_pred_match/num_train_ex))

    # Test this linear spam classifier
    spam_test_data = numpy.genfromtxt("../spamTest.txt", delimiter=",")
    num_test_ex = spam_test_data.shape[0]
    if (num_test_ex == 0): raise Error('num_test_ex == 0')
    x_test_mat = spam_test_data[:, 0:num_features]
    y_test_vec = spam_test_data[:, num_features]
    print("Evaluating the trained Linear SVM on a test set ...")
    pred_vec = svm_model.predict(x_test_mat)
    num_pred_match = 0
    for ex_index in range(0, num_test_ex):
        if (pred_vec[ex_index] == y_test_vec[ex_index]):
            num_pred_match = num_pred_match + 1
    print("Test Accuracy: %.6f" % (100*num_pred_match/num_test_ex))
    input("Program paused. Press enter to continue.")
    print("")

    # Determine the top predictors of spam
    svm_model_weights_sorted = numpy.sort(svm_model.coef_, axis=None)[::-1]
    svm_model_wts_sorted_idx = numpy.argsort(svm_model.coef_, axis=None)[::-1]
    vocab_list = get_vocab_list()
    print("Top predictors of spam:")
    for pred_index in range(0, 15):
        print("%s (%.6f)" % (vocab_list[svm_model_wts_sorted_idx[pred_index]],
                             svm_model_weights_sorted[pred_index]))
    input("Program paused. Press enter to continue.")
    print("")

    # Test this linear spam classifier on another e-mail
    spam_sample_1 = open('../spamSample1.txt', 'r')
    file_contents = []
    while True:
        c = spam_sample_1.read(1)
        if not c:
            break
        else:
            file_contents.append(c)
    spam_sample_1.close()
    email_word_indices = process_email(file_contents)
    features_vec = email_features(email_word_indices)
    pred_vec = svm_model.predict(numpy.transpose(features_vec))
    print("Processed spamSample1.txt")
    print("Spam Classification: %d" % pred_vec[0])
    print("(1 indicates spam, 0 indicates not spam)")

# Call main function
if __name__ == "__main__":
    main()
