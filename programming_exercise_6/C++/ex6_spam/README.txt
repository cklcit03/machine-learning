Note that this C++ code runs rather slowly; the main bottlenecks appear to involve:
1) training the linear SVM classifier
2) assessing the accuracy of this classifier using the training data
3) assessing the accuracy of this classifier using the test data

The training data and test data are 133 MB and 33 MB, respectively.
The Armadillo C++ linear algebra library may have some difficulty loading such large files.

This C++ code utilizes a Porter stemmer algorithm that was written by Sean Massung.
This algorithm can be found at: https://bitbucket.org/smassung/porter2_stemmer/wiki/Home

This C++ code also utilizes LibSVM, which can be found at: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
