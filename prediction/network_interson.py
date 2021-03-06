"""
network_interson.py
~~~~~~~~~~~~~~
This a variation from the network3.py file of Michael Nielsen's book "Neural Networks and Deep Learning"

URL: http://neuralnetworksanddeeplearning.com/

We have added a class call ConvLayer, which performs a Convolution with weights W and bias b.
Also this code contains several learning algorithms, which where extracted from Lasagne github page.

URL: https://github.com/Lasagne/Lasagne
In addition, the code within the SGD class includes Early Stopping, which can be change with the variable tolerance (set by default to 8). And we are saving the training and validation loss for further analysis.

21.03.2016 - New a component to calculate the Specificity and Sensitivity.

22.03.2016 - Contingency table added

23.03.2016 Adding a weight initiation with a Gaussian distribution with std: sqrt(2.0/(n_in* n_out) based on: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" by He et al, 2015 URL: http://arxiv.org/pdf/1502.01852v1.pdf

28.03.2016 Separate the learning functions as a separate library: learning_functions.py. Also types of layers was translated into layer_types.py for better debugging

"""


#### Libraries
# Standard library
import cPickle
import gzip
import math

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
from learning_functions import sgd, apply_nesterov_momentum, nesterov_momentum, get_or_compute_grads

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### Constants
GPU = False
if GPU:
		#print "Trying to run under a GPU.  If this is not desired, then modify "+\
        #"network_interson.py\nto set the GPU flag to False."
    try: 
		theano.config.device = 'gpu'
    except:
		pass # it's already set
		
    theano.config.floatX = 'float64'

class Network():
    
    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")  
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def predict(self,test_data,mini_batch_size = 1):
	   self.test_mb_predictions = theano.function([self.x],self.layers[-1].y_out)
	   predictions = self.test_mb_predictions(test_data)
	   return predictions

#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
