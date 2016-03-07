'''

Script_CNN_interson.py:
This code run the simulation of 64 convolutional neural network with different hyperparameters for further evaluation.
We are running a 17 layer Convolutional Neural Network for Lungs Ultrasound images with 256 x 256 pixel dimensions.
Finally we are saving the network into a Pickle file with the name:
net_l1_deform_<learning>_<lambda>_<mini_batch>_<dropout>.pkl

'''
from datetime import datetime
startTime = datetime.now()
possible_learning_rate = [5.0,3.0,2.0,1.0,1.0/10.0,1.0/100.0,1.0/1000.0,1/10000.0] #8
possible_lambda = [5.0,3.0,2.0,1.0,1.0/10.0,1.0/100.0,1.0/1000.0,1/10000.0] #8
possible_mini_batch = [100]
possible_dropout =0.5
import cPickle
import network_interson
from network_interson import Network
from network_interson import ConvPoolLayer, ConvLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network_interson.load_data_shared()
from network_interson import ReLU
from theano.tensor import tanh
for i in range(len(possible_learning_rate)):
	for j in range(len(possible_lambda)):
		for z in range(len(possible_mini_batch)):
			mini_batch_size = possible_mini_batch[z]
			dropout = possible_dropout
			net = Network([
				ConvLayer(image_shape=(mini_batch_size, 1, 256,256), 
					      filter_shape=(8, 1, 3, 3), 
					      activation_fn=ReLU),
				ConvPoolLayer(image_shape=(mini_batch_size, 8, 254,254), 
					      filter_shape=(8, 8, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
                                ConvPoolLayer(image_shape=(mini_batch_size, 8, 126,126), 
					      filter_shape=(16, 8, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 16, 62,62), 
					      filter_shape=(32, 16, 3, 3), 
					      activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 32, 60,60), 
					      filter_shape=(32, 32, 3, 3), 
					      activation_fn=ReLU),
                                ConvPoolLayer(image_shape=(mini_batch_size, 32, 58,58), 
					      filter_shape=(32, 32, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 32, 28,28), 
					      filter_shape=(64, 32, 3, 3), 
					      activation_fn=ReLU),
                                ConvPoolLayer(image_shape=(mini_batch_size, 64, 26,26), 
					      filter_shape=(64, 64, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 64, 12,12), 
					      filter_shape=(128, 64, 3, 3), 
					      activation_fn=ReLU),
				ConvPoolLayer(image_shape=(mini_batch_size, 128, 10,10), 
					      filter_shape=(128, 128,3, 3), 
					      poolsize=(2, 2),activation_fn = ReLU),
				FullyConnectedLayer(n_in=128*4*4, n_out= 5,activation_fn = ReLU, p_dropout = dropout),
				SoftmaxLayer(n_in=5, n_out=2)], mini_batch_size)
			net.SGD(training_data, 100, mini_batch_size, possible_learning_rate[i],validation_data, test_data,lmbda = possible_lambda[j])
			name = 'net_l1_deform_%(learning)g_%(lambda)g_%(mini_batch)g_%(dropout)g.pkl' %{"learning": possible_learning_rate[i],"lambda":possible_lambda[j],"mini_batch":mini_batch_size,"dropout":dropout}
			f = file(name,'wb')
			cPickle.dump(net,f,protocol=cPickle.HIGHEST_PROTOCOL)
			f.close()
print datetime.now() - startTime
