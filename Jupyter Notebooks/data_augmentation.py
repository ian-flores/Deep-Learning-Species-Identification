import numpy as np
import itertools
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import random
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import flatten
from PIL import Image, ImageOps
from scipy.ndimage.interpolation import shift 
from IPython.display import Image as Im 
from sklearn.utils import shuffle
import sklearn
import pandas

### Linear Augmentation
def linear_augmentation(dataset, num_shifts, labels):
    augmented_dataset = []    
    augmented_labels = []
    for idx, image in enumerate(dataset):
        idx = labels[idx]
        the_image = np.asarray(image)
        for i in range(num_shifts+1):
            pre_image = the_image.reshape((28,28))

            # shift up
            shifted_image_up = shift(pre_image, [(i*(-1)), 0])
            augmented_dataset.append(shifted_image_up)
            augmented_labels.append(idx)
            del shifted_image_up
            
            # shift_down
            shifted_image_down = shift(pre_image, [i, 0]) 
            augmented_dataset.append(shifted_image_down)
            augmented_labels.append(idx)
            del shifted_image_down

            #shift_left
            shifted_image_left = shift(pre_image, [0, (i*(-1))]) 
            augmented_dataset.append(shifted_image_left)
            augmented_labels.append(idx)
            del shifted_image_left

            #shift_right
            shifted_image_right = shift(pre_image, [0, i]) 
            augmented_dataset.append(shifted_image_right)
            augmented_labels.append(idx)
            del shifted_image_right

            del pre_image
        del the_image
    
    return np.asarray(augmented_dataset), np.asarray(augmented_labels)

### Diagonal Augmentation
def diagonal_augmentation(dataset, num_shifts, labels):
    augmented_dataset = []    
    augmented_labels = []
    for idx, image in enumerate(dataset):
        idx = labels[idx]
        the_image = np.asarray(image)
        for i in range(num_shifts+1):
            pre_image = the_image.reshape((28,28))

            # shift diagonal left down
            shifted_image_diagonal_left_down = shift(pre_image, [(i*(-1)), (i*(-1))])
            augmented_dataset.append(shifted_image_diagonal_left_down)
            augmented_labels.append(idx)
            del shifted_image_diagonal_left_down
            
            # shift diagonal right down
            shifted_image_diagonal_right_down = shift(pre_image, [i, (i*(-1))])
            augmented_dataset.append(shifted_image_diagonal_right_down)
            augmented_labels.append(idx)
            del shifted_image_diagonal_right_down
            
            #shift diagonal left up
            shifted_image_diagonal_left_up = shift(pre_image, [(i*(-1)), i])
            augmented_dataset.append(shifted_image_diagonal_left_up)
            augmented_labels.append(idx)
            del shifted_image_diagonal_left_up
            
            # shift diagonal right up
            shifted_image_diagonal_right_up = shift(pre_image, [i, i])
            augmented_dataset.append(shifted_image_diagonal_right_up)
            augmented_labels.append(idx)
            del shifted_image_diagonal_right_up

            del pre_image
        del the_image
    
    return np.asarray(augmented_dataset), np.asarray(augmented_labels)

### Combined Augmentation
def combined_augmentation(dataset, num_shifts, labels):
    augmented_dataset = []    
    augmented_labels = []
    for idx, image in enumerate(dataset):
        idx = labels[idx]
        the_image = np.asarray(image)
        for i in range(num_shifts+1):
            pre_image = the_image.reshape((28,28))

            # shift up
            shifted_image_up = shift(pre_image, [(i*(-1)), 0])
            augmented_dataset.append(shifted_image_up)
            augmented_labels.append(idx)
            del shifted_image_up
            
            # shift_down
            shifted_image_down = shift(pre_image, [i, 0]) 
            augmented_dataset.append(shifted_image_down)
            augmented_labels.append(idx)
            del shifted_image_down

            #shift_left
            shifted_image_left = shift(pre_image, [0, (i*(-1))]) 
            augmented_dataset.append(shifted_image_left)
            augmented_labels.append(idx)
            del shifted_image_left

            #shift_right
            shifted_image_right = shift(pre_image, [0, i]) 
            augmented_dataset.append(shifted_image_right)
            augmented_labels.append(idx)
            del shifted_image_right
            
            # shift diagonal left down
            shifted_image_diagonal_left_down = shift(pre_image, [(i*(-1)), (i*(-1))])
            augmented_dataset.append(shifted_image_diagonal_left_down)
            augmented_labels.append(idx)
            del shifted_image_diagonal_left_down
            
            # shift diagonal right down
            shifted_image_diagonal_right_down = shift(pre_image, [i, (i*(-1))])
            augmented_dataset.append(shifted_image_diagonal_right_down)
            augmented_labels.append(idx)
            del shifted_image_diagonal_right_down
            
            #shift diagonal left up
            shifted_image_diagonal_left_up = shift(pre_image, [(i*(-1)), i])
            augmented_dataset.append(shifted_image_diagonal_left_up)
            augmented_labels.append(idx)
            del shifted_image_diagonal_left_up
            
            # shift diagonal right up
            shifted_image_diagonal_right_up = shift(pre_image, [i, i])
            augmented_dataset.append(shifted_image_diagonal_right_up)
            augmented_labels.append(idx)
            del shifted_image_diagonal_right_up

            del pre_image
        del the_image
    
    return np.asarray(augmented_dataset), np.asarray(augmented_labels)

### CNN Reformat
def reformat(dataset):
    dataset = dataset.reshape((-1, 28, 28, 1, )).astype(np.float32)
    return dataset

### SVM Reformat
def svm_reformat(dataset):
    dataset = dataset.reshape((len(dataset), -1)).astype(np.float32)
    return dataset


### CNN Evaluate
def evaluate(X_data, y_data):
    sess = tf.get_default_session()
    accuracy = sess.run(accuracy_operation, feed_dict={x: X_data, y: y_data})
    return accuracy

### CNN Le-Net5
def LeNet(x):    
    mu = 0
    sigma = 0.1
    
    W = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean = mu, stddev = sigma))
    b = tf.Variable(tf.zeros(6))
    layer1 = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="VALID")
    layer1 = tf.nn.bias_add(layer1, b)
    layer1 = tf.nn.relu(layer1)#conv2d(x, W, b, 1, 'VALID')
    pool1 = tf.nn.max_pool(layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

    W = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean = mu, stddev = sigma))
    b = tf.Variable(tf.zeros(16))
    layer2 = tf.nn.conv2d(pool1, W, strides=[1,1,1,1], padding="VALID")
    layer2 = tf.nn.bias_add(layer2, b)
    layer2 = tf.nn.relu(layer2)#conv2d(x, W, b, 1, 'VALID')
    pool2 = tf.nn.max_pool(layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
    
    fc = flatten(pool2)
    
    W = tf.Variable(tf.truncated_normal([400, 120], mean = mu, stddev = sigma))
    b = tf.Variable(tf.zeros(120))
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc, W), b))

    W = tf.Variable(tf.truncated_normal([120, 84], mean = mu, stddev = sigma))
    b = tf.Variable(tf.zeros(84))
    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, W), b))
    
    W = tf.Variable(tf.truncated_normal([84, 21], mean = mu, stddev = sigma))
    b = tf.Variable(tf.zeros(21))
    logits = tf.add(tf.matmul(fc2, W), b)
    
    return logits

