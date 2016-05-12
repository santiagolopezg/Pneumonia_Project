'''
Implementation of cross validation for CNN, FC and logistic regression. Returns average sensitivity, specificity, F1 score and MCC.
'''

import cPickle
CNNsens,CNNspec,CNNF1,CNNmcc = [0,0,0,0]
CNNparams = [CNNsens,CNNspec,CNNF1,CNNmcc]
FCsens,FCspec,FCF1,FCmcc = [0,0,0,0]
FCparams = [FCsens,FCspec,FCF1,FCmcc]
LRsens,LRspec,LRF1,LRmcc = [0,0,0,0]
LRparams = [LRsens,LRspec,LRF1,LRmcc]
for i in xrange(9):
	f = open('cnn_metrics_{0}.pkl'.format(i+1),'rb')
	g = open('fullyconnected_metrics_{0}.pkl'.format(i+1),'rb')
	h = open('logisticregression_metrics_{0}.pkl'.format(i+1),'rb')
	CNN = cPickle.load(f)
	f.close()
	CNNparams[0] += CNN[0]
	CNNparams[1] += CNN[1]
	CNNparams[2] += CNN[2]
	CNNparams[3] += CNN[3]

	FC = cPickle.load(g)
	g.close()
	FCparams[0] += FC[0]
	FCparams[1] += FC[1]
	FCparams[2] += FC[2]
	FCparams[3] += FC[3]

	LR = cPickle.load(h)
	h.close()
	LRparams[0] += LR[0]
	LRparams[1] += LR[1]
	LRparams[2] += LR[2]
	LRparams[3] += LR[3]

for j in xrange(4):
	try:
		CNNparams[j] = CNNparams[j]/9.0
	except ZeroDivisionError:
		CNNparams[j] = 0
		print 'Nuuuuuuu'
	try:
		FCparams[j] = FCparams[j]/9.0
	except ZeroDivisionError:
		FCparams[j] = 0
		print 'Nuuuuuuu'
	try:
		LRparams[j] = LRparams[j]/9.0
	except ZeroDivisionError:
		LRparams[j] = 0
		print 'Nuuuuuuu'

for v in xrange(4):
	CNNparams[v] = round(CNNparams[v], 4)
	FCparams[v] = round(FCparams[v], 4)
	LRparams[v] = round(LRparams[v], 4)

print 'Cross validation results for CNN (sens, spec, F1, mcc): ', CNNparams
print 'Cross validation results for Fully Connected network (sens, spec, F1, mcc): ', FCparams
print 'Cross validation results for logistic regression (sens, spec, F1, mcc): ',LRparams
