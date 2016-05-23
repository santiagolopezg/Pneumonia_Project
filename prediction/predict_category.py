########################################
def prediction_category(i):
	import cPickle
	import network_interson
	from network_interson import Network
	from layer_types import ConvPoolLayer, ConvLayer, FullyConnectedLayer, SigmoidLayer, SoftmaxLayer
	from network_interson import ReLU
	from theano.tensor import tanh
	from scipy import ndimage
	import numpy as np
	import theano

	########################################
	#Get data

	file_name = "parameters_0.001_0.001_100_0.5.pkl"
	wei = open(file_name,'rb')
	weigh = cPickle.load(wei)
	wei.close()
	weights = []
	for i in weigh:
		weights.append(i.get_value())

	net = Network([ConvLayer(image_shape=(1, 1, 256,256), filter_shape=(8, 1, 3, 3),activation_fn = ReLU, w = weights[0], b = weights[1]), ConvPoolLayer(image_shape=(1, 8, 254,254), 
	filter_shape=(8, 8, 3, 3),poolsize=(2, 2), activation_fn = ReLU, w = weights[2], b = weights[3]),ConvPoolLayer(image_shape=(1, 8, 126,126), 
	filter_shape=(8, 8, 3, 3), poolsize=(2, 2), activation_fn = ReLU, w = weights[4], b = weights[5]),ConvLayer(image_shape=(1, 8, 62,62),filter_shape=(16, 8, 3, 3), 
	activation_fn = ReLU, w = weights[6], b = weights[7]),ConvLayer(image_shape=(1, 16, 60,60),	      filter_shape=(16, 16, 3, 3), activation_fn = ReLU, w = weights[8], b = weights[9]),ConvPoolLayer(image_shape=(1, 16, 58,58),	      filter_shape=(16, 16, 3, 3), poolsize=(2, 2), activation_fn = ReLU, w = weights[10], b = weights[11]),ConvLayer(image_shape=(1, 16, 28,28), 	      filter_shape=(32, 16, 3, 3),		      activation_fn=ReLU, w = weights[12],b = weights[13]),ConvPoolLayer(image_shape=(1, 32, 26,26),	      filter_shape=(32, 32, 3, 3), poolsize=(2, 2), activation_fn=ReLU, w = weights[14],b = weights[15]),ConvLayer(image_shape=(1, 32, 12,12),     filter_shape=(32, 32, 3, 3),	      activation_fn=ReLU, w = weights[16],b = weights[17]), ConvPoolLayer(image_shape=(1, 32, 10,10), 	      filter_shape=(32, 32,3, 3), poolsize=(2, 2),activation_fn = ReLU, w = weights[18],b = weights[19]),
					FullyConnectedLayer(n_in=32*4*4, n_out= 10,activation_fn = ReLU, w = weights[20],b = weights[21], p_dropout = 0.0),
					FullyConnectedLayer(n_in=10, n_out= 5,activation_fn = ReLU, w = weights[22],b = weights[23], p_dropout = 0.0),
					SoftmaxLayer(n_in=5, n_out=2, w = weights[24],b = weights[25])], 1)


	image_new = ndimage.imread(i)
	image_new = np.reshape(image_new,(1,256*256))
	image = theano.shared(np.asarray(image_new, dtype=theano.config.floatX), borrow=True)
	a = net.predict(image_new)
	return a
