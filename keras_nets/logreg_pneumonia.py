'''Train a simple convnet on the MNIST dataset.

Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

Get to 99.25% val accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
import h5py
import keras
import cPickle
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, l1l2, l1
import math
import sklearn

def F1_score(y_true,y_pred):
	f1 = sklearn.metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
	return f1

def load_data_p(number):
	'''
	This code lodes the data from the Pneumonia
	dataset into the model
	'''
	
	f = open('neumonia_dataset_interson_keras_alldata10_{0}.pkl'.format(number),'rb')
	data = cPickle.load(f)
	f.close()
	training_data = data[0]
	validation_data = data[1]
	test_data = data[2]

	label_t = training_data[1]
	data_t = training_data[0]

	label_v = validation_data[1]
	data_v = validation_data[0]
	
	label_tt = test_data[1]
	data_tt = test_data[0]

	return (data_t,label_t),(data_v,label_v),(data_tt,label_tt)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))



batch_size = 100
nb_classes = 2
nb_epoch = 70

# input image dimensions
img_rows, img_cols = 256, 256
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#Dataset to use
for jk in xrange(9):
	number_db = jk + 1
	# the data, shuffled and split between tran and val sets
	(X_train, y_train), (X_val, y_val), (X_test,y_test) = load_data_p(number_db)


	#X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)

	#X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)
	#X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

	X_train = X_train.astype('float32')
	X_val = X_val.astype('float32')
	X_test = X_test.astype('float32')

	#X_train /= 255
	#X_val /= 255
	#X_test /=255
	print(X_train.shape)
	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_val.shape[0], 'val samples')
	print(X_test.shape[0], 'test samples')

	model = Sequential()
	model.add(Dense(output_dim=1, W_regularizer=keras.regularizers.l2(1.),input_dim=(256*256)))
	model.add(Activation("sigmoid"))

	sgd = SGD(lr=0.1)
	model.compile(loss='binary_crossentropy',class_mode='binary', optimizer=sgd)

	#Implement Early Stopping and safe loss history for plotting

	early_stopping = EarlyStopping(monitor='val_loss', patience=15)
	history = LossHistory()

	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
		  show_accuracy=True, verbose=1,validation_data=(X_val, y_val),callbacks=[early_stopping,history],shuffle=True)
	#Try to get sensitivity

	prediction = model.predict(X_test,batch_size=y_test.shape[0])

	prediction[prediction > 0.5] = 1.0
	prediction[prediction <= 0.5] = 0.0

	F1 = 0.0
	mcc = 0.0

	TP = 0
	for i in xrange(len(y_test)):
		if y_test[i] == 1.0 and prediction[i] == 1.0:
			TP+=1.0

		

	TN = 0
	for i in xrange(len(y_test)):
		if y_test[i]  == 0.0 and prediction[i] == 0.0:
			TN+= 1.0


	FP = 0
	for i in xrange(len(y_test)):
		if y_test[i] < prediction[i]:
			FP+= 1.0


	FN = 0
	for i in xrange(len(y_test)):
		if y_test[i] > prediction[i]:
			FN+= 1.0


	sensitivity = TP/(TP + FN)
	specificity = TN/(TN + FP)

	print (sensitivity, specificity)

	print('Test Sensitivity Score: {0:.2%}'.format(sensitivity))
	print('Test Specificity Score: {0:.2%}'.format(specificity))

	try:
		PPV = TP / (TP + FP)
		NPV = TN / (TN + FN)
		F1 = 2.0 * (PPV * sensitivity)/(PPV + sensitivity)


		mcc = (TP*TN - FP*FN)/(math.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)))
		print('Test F1 Score: {0:.2%}'.format(F1))

	

	except ZeroDivisionError:
		print('Divide by Zerapio')
	score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	#Safe arquitecture of my model

	json_string = model.to_json()

	'''# model reconstruction from JSON:
	from keras.models import model_from_json
	model = model_from_json(json_string)'''

	#Safe weights
	name = 'neumonia_dataset_interson_keras_alldata_{0}_weights_logistic_{0}.h5'.format(number_db,(jk+1))
	print(name)
	model.save_weights(name,overwrite=True)

	#l = h5py.File("loss_history_{0}.hdf5".format(number_db), "w")
	#dset = f.create_dataset("loss_history_{0}".format(number_db), (100,), dtype='i')


	#A way to open a model with weights in the same arquitecture
	'''
	json_string = model.to_json()
	open('my_model_architecture.json', 'w').write(json_string)
	model.save_weights('my_model_weights.h5')
	'''
	import cPickle
	f = open('logisticregression_loss_{0}.pkl'.format(jk+1),'wb')
	cPickle.dump(history.losses,f,protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()
	
	h = open('logisticregression_metrics_{0}.pkl'.format(jk+1),'wb')
	
	cPickle.dump([sensitivity,specificity,F1,mcc],h,protocol=cPickle.HIGHEST_PROTOCOL)
	h.close()
	model.reset_states()

	#import matplotlib.pylab as plt
	#plt.plot(history.losses,'bo')
	#plt.xlabel('Iteration')
	#plt.ylabel('Binary Cross Entropy')
	#plt.show()

