# Pneumonia_Project

This is a project from the Bioinformatics Unit at Universidad Peruana Cayetano Heredia, for Biomedical image classification.

We will use a Convolutional Neural Network to classify ultrasound images of patients, as presenting or not presenting pneumonia.


**Collaborators:**
 * Josue Ortega Caro 
 * Santiago Lopez 
 * Franklin Barrientos 
 * Ronald Barrientos 


##Content

* install_cudas is a txt file with intructions on how to configure Theano with a fresh install of Ubuntu 14.04 and a TITAN X GPU. These instructions have been tested in 5 computers.

* The folders CNN, Fully Connected and Logistic regression contain the implementation for 3 different classifiers for comparison. This code was written in Theano, which is a Python libraty for Deep Learning. This code is legacy but functional.

* The folder keras_net is the most recent implementation and the one being used for the manuscript. This code was written using the Keras wrapper for Theano and cross_validation_sklearn.py uses the Scikit-learn API for optimization and cross validation for hyperparameter search


## Licence

Copyright (c) 2015-2016 Josue Ortega Caro

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
