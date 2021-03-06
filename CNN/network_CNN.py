from datetime import datetime
########################################
# Look for the data at:

file_name = "../data/normal_shuffle/neumonia_dataset_interson_elDeform_0_2.pkl"
########################################
# HyperParameters
startTime = datetime.now()
possible_learning_rate = [1,1.0/10.0,1.0/100.0,1.0/1000.0,1/10000.0] #5
#possible_learning_rate = [1/10000.0] #5
possible_lambda = [5.0,2.0,3.0,1.0,1.0/10.0,1.0/100.0,1.0/1000.0] #7
#possible_lambda = [1.0] #7
possible_mini_batch = [100]
possible_dropout =0.5
########################################
# Import libraries
import cPickle
import network_interson
from network_interson import Network
from layer_types import ConvPoolLayer, ConvLayer, FullyConnectedLayer, SigmoidLayer, SoftmaxLayer
training_data, validation_data, test_data = network_interson.load_data_shared(filename= file_name)
from network_interson import ReLU
from theano.tensor import tanh

########################################
#Actual training and research of Hyperparameters

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
					      filter_shape=(8, 8, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 8, 62,62), 
					      filter_shape=(16, 8, 3, 3), 
					      activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 16, 60,60), 
					      filter_shape=(16, 16, 3, 3), 
					      activation_fn=ReLU),
                                ConvPoolLayer(image_shape=(mini_batch_size, 16, 58,58), 
					      filter_shape=(16, 16, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 16, 28,28), 
					      filter_shape=(32, 16, 3, 3), 
					      activation_fn=ReLU),
                                ConvPoolLayer(image_shape=(mini_batch_size, 32, 26,26), 
					      filter_shape=(32, 32, 3, 3), 
					      poolsize=(2, 2), activation_fn=ReLU),
				ConvLayer(image_shape=(mini_batch_size, 32, 12,12), 
					      filter_shape=(32, 32, 3, 3), 
					      activation_fn=ReLU),
				ConvPoolLayer(image_shape=(mini_batch_size, 32, 10,10), 
					      filter_shape=(32, 32,3, 3), 
					      poolsize=(2, 2),activation_fn = ReLU),
				FullyConnectedLayer(n_in=32*4*4, n_out= 10,activation_fn = ReLU, p_dropout = dropout),
				FullyConnectedLayer(n_in=10, n_out= 5,activation_fn = ReLU, p_dropout = dropout),
				SoftmaxLayer(n_in=5, n_out=2)], mini_batch_size)
			net.SGD(training_data, 50, mini_batch_size, possible_learning_rate[i],validation_data, test_data,lmbda = possible_lambda[j])
			name = 'net_normal0_%(learning)g_%(lambda)g_%(mini_batch)g_%(dropout)g.pkl' %{"learning": possible_learning_rate[i],"lambda":possible_lambda[j],"mini_batch":mini_batch_size,"dropout":dropout}
			f = file(name,'wb')
			cPickle.dump(net,f,protocol=cPickle.HIGHEST_PROTOCOL)
			f.close()
print datetime.now() - startTime
