########################################

class network_p():


	def __init__(self):

		self.cPickle = __import__('cPickle')
		self.Network = __import__('network_interson').Network
		self.ConvPoolLayer = __import__('layer_types').ConvPoolLayer
		self.ConvLayer = __import__('layer_types').ConvLayer
		self.FullyConnectedLayer = __import__('layer_types').FullyConnectedLayer
		self.SoftmaxLayer = __import__('layer_types').SoftmaxLayer
		self.ReLU = __import__('network_interson').ReLU
		self.ndimage = __import__('scipy').ndimage
		self.np = __import__('numpy')
		self.imresize = __import__('scipy').misc.imresize
		########################################
		#Get data

		file_name = "params_0.001_0.001_100_0.5.pkl"
		wei = open(file_name,'rb')
		weights = self.cPickle.load(wei)
		wei.close()

		self.net = self.Network([self.ConvLayer(image_shape=(1, 1, 256,256), filter_shape=(8, 1, 3, 3),activation_fn = self.ReLU, w = weights[0], b = weights[1]), self.ConvPoolLayer(image_shape=(1, 8, 254,254), filter_shape=(8, 8, 3, 3),poolsize=(2, 2), activation_fn = self.ReLU, w = weights[2], b = weights[3]),self.ConvPoolLayer(image_shape=(1, 8, 126,126), filter_shape=(8, 8, 3, 3), poolsize=(2, 2), activation_fn = self.ReLU, w = weights[4], b = weights[5]),self.ConvLayer(image_shape=(1, 8, 62,62),filter_shape=(16, 8, 3, 3), activation_fn = self.ReLU, w = weights[6], b = weights[7]),self.ConvLayer(image_shape=(1, 16, 60,60), filter_shape=(16, 16, 3, 3), activation_fn = self.ReLU, w = weights[8], b = weights[9]),self.ConvPoolLayer(image_shape=(1, 16, 58,58),filter_shape=(16, 16, 3, 3), poolsize=(2, 2), activation_fn = self.ReLU, w = weights[10], b = weights[11]),self.ConvLayer(image_shape=(1, 16, 28,28), 	      filter_shape=(32, 16, 3, 3), activation_fn=self.ReLU, w = weights[12],b = weights[13]),self.ConvPoolLayer(image_shape=(1, 32, 26,26),	      filter_shape=(32, 32, 3, 3), poolsize=(2, 2), activation_fn=self.ReLU, w = weights[14],b = weights[15]),self.ConvLayer(image_shape=(1, 32, 12,12),     filter_shape=(32, 32, 3, 3),	      activation_fn=self.ReLU, w = weights[16],b = weights[17]), self.ConvPoolLayer(image_shape=(1, 32, 10,10), 	      filter_shape=(32, 32,3, 3), poolsize=(2, 2),activation_fn = self.ReLU, w = weights[18],b = weights[19]),self.FullyConnectedLayer(n_in=32*4*4, n_out= 10,activation_fn = self.ReLU, w = weights[20],b = weights[21], p_dropout = 0.0),self.FullyConnectedLayer(n_in=10, n_out= 5,activation_fn = self.ReLU, w = weights[22],b = weights[23], p_dropout = 0.0),self.SoftmaxLayer(n_in=5, n_out=2, w = weights[24],b = weights[25])], 1)

	def prediction(self,i):
		image_new = self.imresize(i,(256,256))
		image_new = self.np.reshape(image_new,(1,256*256))
		a = self.net.predict(image_new)
		return a[0]
