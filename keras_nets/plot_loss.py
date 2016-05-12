import cPickle
import matplotlib.pylab as plt

f = open('cnn_loss.pkl','rb')
cnn_loss = cPickle.load(f)
f.close()

f = open('fullyconnected_loss.pkl','rb')
fc_loss = cPickle.load(f)
f.close()

f = open('logisticregression_loss.pkl','rb')
lr_loss = cPickle.load(f)
f.close()

plt.plot(cnn_loss, 'ro', label='Convolutional Neural Network')
plt.plot(fc_loss, 'bo',label='Fully Connected Network')
plt.plot(lr_loss, 'ko',label='Logistic Regression')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
