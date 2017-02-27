# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

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

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, padding='SAME'):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

batch_size = 128
patch_size = 5
depth1 = 6
depth2 = 16
num_hidden1 = 120
num_hidden2 = 84

beta_reg = 3e-5

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  tf_regulariztion = tf.placeholder(tf.float32)
  
  # Variables.
  layer1_weights = weight_variable([patch_size, patch_size, num_channels, depth1])
  layer1_biases = bias_variable([depth1])


  layer2_weights = weight_variable([patch_size, patch_size, depth1, depth2])
  layer2_biases = bias_variable([depth2])

  # size3 = ((image_size - patch_size + 1) // 2 - patch_size + 1) // 2
  input_layer_dim = image_size//4 - 2;
  layer3_weights = weight_variable(
  	[input_layer_dim * input_layer_dim * depth2, num_hidden1])
  layer3_biases = bias_variable([num_hidden1])

  layer4_weights = weight_variable([num_hidden1, num_hidden2])
  layer4_biases = bias_variable([num_hidden2])

  layer5_weights = weight_variable([num_hidden2, num_labels])
  layer5_biases = bias_variable([num_labels])

  pool_layer_bias1 = bias_variable([depth1])
  pool_layer_bias2 = bias_variable([depth2])
  
  # Model.
  def model(data, keep_prob): 
    hidden = tf.nn.relu(conv2d(data, layer1_weights) + layer1_biases)
    print ('conv layer 1: ' + str(hidden.get_shape()))
    
    pool = tf.nn.relu(max_pool_2x2(hidden) + pool_layer_bias1)
    print ('pool layer 2: ' + str(pool.get_shape()))
    
    hidden = tf.nn.relu(conv2d(pool, layer2_weights, padding='VALID')+layer2_biases)
    print ('conv layer 3: ' + str(hidden.get_shape()))
    
    pool = tf.nn.relu(max_pool_2x2(hidden) + pool_layer_bias2)
    print ('pool layer 4: ' + str(pool.get_shape()))
    
    shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
    print ('reshaped layer: ' + str(reshape.get_shape()))
    
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    print ('fc layer: ' + str(hidden.get_shape()))
    
    dropout = tf.nn.dropout(hidden, keep_prob)
    
    hidden = tf.nn.relu(tf.matmul(dropout, layer4_weights) + layer4_biases)
    print ('fc layer: ' + str(hidden.get_shape()))
    
    return tf.matmul(hidden, layer5_weights) + layer5_biases
  
  # Training computation.
  droput_keep_prob = 1.0
  logits = model(tf_train_dataset, droput_keep_prob)

  # L2 Regulariztion Loss
  l2_loss = tf_regulariztion*tf.nn.l2_loss(layer1_weights) +\
    tf_regulariztion*tf.nn.l2_loss(layer2_weights) +\
    tf_regulariztion*tf.nn.l2_loss(layer3_weights) +\
    tf_regulariztion*tf.nn.l2_loss(layer4_weights) +\
    tf_regulariztion*tf.nn.l2_loss(layer5_weights)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=tf_train_labels) + l2_loss)
    
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
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0))
  test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))


num_steps = 6001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,
    			 tf_regulariztion : beta_reg}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))