import network_interson_2
import cPickle
import matplotlib.pylab as plt

f = open('net_trial_0.001_0.1_200_0.5.pkl','rb')
net = cPickle.load(f)
f.close()

cost_train = net.cost_train
cost_val = net.cost_validation
plt.plot(cost_train,'ro',label='Training')
plt.plot(cost_val,'bo',label='Validation')
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Cost Function')
plt.legend(loc = 'upper right')
plt.show()
