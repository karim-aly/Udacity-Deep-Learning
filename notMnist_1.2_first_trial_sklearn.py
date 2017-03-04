# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Loading datasets from the saved pickle file

pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
	datasets = pickle.load(f)

train_dataset = datasets['train_dataset']
train_labels = datasets['train_labels']
valid_dataset = datasets['valid_dataset']
valid_labels = datasets['valid_labels']
test_dataset = datasets['test_dataset']
test_labels = datasets['test_labels']

# Take number of samples from dataset
nsamples_all = [10, 100, 1000, 5000]
for nsamples in nsamples_all:
    train_dataset_n = train_dataset[0:nsamples, :, :]
    train_labels_n = train_labels[0:nsamples,]

    test_dataset_n = test_dataset[0:nsamples, :, :]
    test_labels_n = test_labels[0:nsamples,]

    # Train the Logistic Regression Model

    nsamples, nx, ny = train_dataset_n.shape
    d2_train_dataset = train_dataset_n.reshape((nsamples,nx*ny))

    nsamples, nx, ny = test_dataset_n.shape
    d2_test_dataset = test_dataset_n.reshape((nsamples,nx*ny))

    logistic = LogisticRegression()
    logistic.fit(d2_train_dataset, train_labels_n)

    # Calc. the mean accuracy of the model on the test data

    score = logistic.score(d2_test_dataset, test_labels_n)

    # Print Results

    print('For %d Training Example, LogisticRegression score: %f' % (nsamples, score))
