import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import theano.tensor as T
import lasagne
from lasagne import init
from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import DenseLayer, InputLayer, FeaturePoolLayer
from lasagne.updates import adam
from nolearn.lasagne import NeuralNet

random.seed(10)
L1 = 0.1
L2 = 0.1
B = 25000
N_HIDDEN_1 = 250
N_HIDDEN_2 = 150
N_EPOCHS = 20

x_train = np.load('../input/Train.npy').astype(np.float32)
x_test = np.load('../input/Test.npy').astype(np.float32)
y_train = np.load('../input/Label.npy')
id_test = np.load('../input/ID.npy')
print x_train.shape, x_test.shape, y_train.shape

print "Data is critically imblanced. The number of 1's and 0's are {} and {}, respectively"\
    .format(np.sum(y_train==1), np.sum(y_train==0))

ind = np.where(y_train==1)[0]
prob = np.ones(len(ind))/len(ind)
boot_sample = np.random.choice(ind, size=B, p=prob)
print "New dataset using bootstrap"
x_train = np.concatenate((x_train, x_train[boot_sample]), axis=0)
y_train = np.concatenate((y_train, y_train[boot_sample]), axis=0).astype(np.int32)
n_train = x_train.shape[1]
new_ind = np.random.permutation(n_train)
x_train = x_train[new_ind]
y_train = y_train[new_ind]
print x_train.shape, x_test.shape, y_train.shape

def custom_objective(predictions, targets):
    loss = lasagne.objectives.binary_accuracy(predictions, targets)
    thrid_layer = net.layers_
    loss += L1 * lasagne.regularization.regularize_network_params(thrid_layer['dense1'], lasagne.regularization.l1)  
    loss += L2 * lasagne.regularization.regularize_network_params(thrid_layer['dense1'], lasagne.regularization.l2)
    loss += L2 * lasagne.regularization.regularize_network_params(thrid_layer['dense3'], lasagne.regularization.l2)
    return loss

def NEURAL_NET(n_features, labels, eval_size=0.20):
    
    layers = [
        (InputLayer, {'shape': (None, n_features)}),
        (DenseLayer, {'num_units': N_HIDDEN_1, 'nonlinearity': rectify,
                      'W': init.Orthogonal('relu'),
                      'b': init.Constant(0.01)}),
        (FeaturePoolLayer, {'pool_size': 2}),
        (DenseLayer, {'num_units': N_HIDDEN_2, 'nonlinearity': rectify,
                      'W': init.Orthogonal('relu'),
                      'b': init.Constant(0.01)}),
        (FeaturePoolLayer, {'pool_size': 2}),
        (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
    ]    
    args = dict(
        update=adam,
        regression=False,
        eval_size=eval_size,
        max_epochs=N_EPOCHS,
        verbose=1,
    )
    net = NeuralNet(layers, **args)
    return net

scaler= StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)
'''
global net
net = NEURAL_NET(n_features = x_train.shape[1], labels = y_train, eval_size=0.10)
net.fit(x_train, y_train)
print(roc_auc_score(y_train, net.predict_proba(x_train)[:,1]))

net = NEURAL_NET(n_features = x_train.shape[1], labels = y_train, eval_size=None)
net.fit(x_train, y_train)
# predicting
y_pred= net.predict_proba(x_test)[:,1]

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("../input/submission.csv", index=False)
'''
print('Completed!')





