# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt

# First reload the data we generated in 1_notmnist.ipynb.

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

# Reformat into a shape that's more adapted to the models we're going to train:
#   data as a flat matrix,
#   labels as float 1-hot encodings.

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

"""
We're first going to train a multinomial logistic regression using simple 
gradient descent.

TensorFlow works like this:
First you describe the computation that you want to see performed: 
  what the inputs, the variables, and the operations look like.

These get created as nodes over a computation graph. 
This description is all contained within the block below:

	with graph.as_default():
	    ...

Then you can run the operations on this graph as many times as you want 
by calling session.run(), providing it outputs to fetch from the graph 
that get returned. This runtime operation is all contained in the block below:

	with tf.Session(graph=graph) as session:
	    ...

Let's load all the data into TensorFlow and build the computation 
graph corresponding to our training.
"""

batch_size = 128

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  tf_regulariztion = tf.placeholder(tf.float32)
  #tf_regulariztion = tf.constant(beta_reg, dtype='float32')
  
  # Variables.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases

  # L2 Regulariztion Loss
  l2_loss = tf_regulariztion*tf.nn.l2_loss(weights)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=tf_train_labels) + l2_loss)
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

# Let's run this computation and iterate

num_steps = 3001

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

beta_range = [pow(10, i) for i in np.arange(-4, -2, 0.1)]
valid_accuracy_val = []
test_accuracy_val = []

for beta_reg in beta_range:
  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Beta = %f" % beta_reg)
    for step in range(num_steps):
      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      # Generate a minibatch.
      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,
                  tf_regulariztion : beta_reg}
      _, l, predictions = session.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)
    valid_accuracy_val.append(accuracy(valid_prediction.eval(), valid_labels))
    test_accuracy_val.append(accuracy(test_prediction.eval(), test_labels))
    print("Validation accuracy: %.1f%%" % valid_accuracy_val[-1])
    print("Test accuracy: %.1f%%" % test_accuracy_val[-1])


plt.semilogx(beta_range, valid_accuracy_val)
plt.grid(True)
plt.title('Validation accuracy by regularization (logistic)')
plt.show()

plt.figure()
plt.semilogx(beta_range, test_accuracy_val)
plt.grid(True)
plt.title('Test accuracy by regularization (logistic)')
plt.show()

# We choose the beta_req to be 0.0015