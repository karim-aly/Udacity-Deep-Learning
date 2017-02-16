# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

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
hidden_layer_nodes1 = 1024
hidden_layer_nodes2 = 300
beta_reg = 2e-4

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
  hidden_weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_layer_nodes1],
        stddev=np.sqrt(2.0 / (image_size * image_size))))
  hidden_biases1 = tf.Variable(tf.zeros([hidden_layer_nodes1]))

  hidden_weights2 = tf.Variable(
    tf.truncated_normal([hidden_layer_nodes1, hidden_layer_nodes2],
        stddev=np.sqrt(2.0 / hidden_layer_nodes1)))
  hidden_biases2 = tf.Variable(tf.zeros([hidden_layer_nodes2]))

  # Out weights connect hidden neurons to output labels
  out_weights = tf.Variable(
    tf.truncated_normal([hidden_layer_nodes2, num_labels],
        stddev=np.sqrt(2.0 / hidden_layer_nodes2)))
  out_biases = tf.Variable(tf.zeros([num_labels]))

  # Hidden ReLU Activated Layer
  hidden_layer1 = tf.nn.relu(tf.matmul(
    tf_train_dataset, hidden_weights1) + hidden_biases1)

  # Hidden ReLU Activated Layer
  hidden_layer2 = tf.nn.relu(tf.matmul(
    hidden_layer1, hidden_weights2) + hidden_biases2)
  
  # Training computation.
  logits = tf.matmul(hidden_layer2, out_weights) + out_biases

  # L2 Regulariztion Loss
  l2_loss = tf_regulariztion*tf.nn.l2_loss(hidden_weights1) +\
    tf_regulariztion*tf.nn.l2_loss(hidden_weights2) +\
    tf_regulariztion*tf.nn.l2_loss(out_weights)

  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + l2_loss)
  
  # Optimizer. (Gradient Descent with exponential decay learning rate)
  global_step = tf.Variable(0, trainable=False)  # count the number of steps taken
  starter_learning_rate = 0.35
  learning_rate = tf.train.exponential_decay(
    starter_learning_rate, global_step, 500, 0.85, staircase=True)

  # Passing global_step to minimize() will increment it at each step.
  optimizer = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)

  valid_hidden_layer1 = tf.nn.relu(
    tf.matmul(tf_valid_dataset, hidden_weights1) + hidden_biases1)
  valid_hidden_layer2 = tf.nn.relu(
    tf.matmul(valid_hidden_layer1, hidden_weights2) + hidden_biases2)
  valid_prediction = tf.nn.softmax(
    tf.matmul(valid_hidden_layer2, out_weights) + out_biases)
  
  test_hidden_layer1 = tf.nn.relu(
    tf.matmul(tf_test_dataset, hidden_weights1) + hidden_biases1)
  test_hidden_layer2 = tf.nn.relu(
    tf.matmul(test_hidden_layer1, hidden_weights2) + hidden_biases2)
  test_prediction = tf.nn.softmax(
    tf.matmul(test_hidden_layer2, out_weights) + out_biases)

# Let's run this computation and iterate

num_steps = 9001

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run() # 'initialize_all_variables' deprecated
  print("Initialized")
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
    if (step % 500 == 0):
      print("Learning rate: %f" % learning_rate.eval())
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

  # Last Run Output:
  # Test accuracy: 95.9%