# CAFIR-10-classification-by-HOG-and-histogram-features-with-MLP
This code is a project from DIP course.
The aim of project is to extract any features you like to help a classsifier to finish CAFIR-10 dataset classification task, the required accuracy of which is above 54%.

## description
Our implements include about three parts:
### data processing
load data from dataset, reshape them to satisfied form for feature extracting.
this part is in data_utils.py
### feature extracting
use hog_feature.py and color_histogram_hsv.py to extract HOG and histogram features.
the features are stored as txt files, i.e. hsv_feather_Xtest.txt, hsv_feather_Xtrain.txt and histogram_feather_Xtest.txt, histogram_feather_Xtrain.txt.
### classification
I have construct two classifier in my_classifier.py, randomForest and MLP classifiers, with the help of sklearn library.

## TODO
more classifier can be used to get higher accuracy
more features can be used for train models,such as SIFT, here is a good open source code for extracting SIFT feature from figures, https://github.com/rmislam/PythonSIFT.

## Reference
https://blog.csdn.net/HHH_ANS/article/details/86297859
