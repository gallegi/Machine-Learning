import load_data
import time
import numpy as np

import tensorflow as tf
from  tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import dropout

batch_size = 128
n_classes = 43
classes = []

n_epochs = 101
print_every_n_epochs = 10

is_training = tf.placeholder(tf.bool, shape=(), name= 'is_training')
bn_params = {'is_training': is_training,
                 'decay': 0.99,
                 'updates_collections': None}

input_channels = 3
conv1_ksize = 3
conv1_n_filters = 32

conv2_ksize = 3
conv2_n_filters = 64

conv3_ksize = 3
conv3_n_filters = 128

def create_weights(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape, stddev= 0.05), dtype=tf.float32)

def create_bias(size):
    return tf.Variable(initial_value=0.05, dtype=tf.float32)

def create_convolutional_layer(input, n_input_channels, filter_size, n_filters):

    # create filters
    weights = create_weights(shape=(filter_size, filter_size, n_input_channels, n_filters))

    # create conv2d layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='VALID')

    # add biases term
    layer += create_bias(n_filters)

    # add max pooling layer
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # add activation function
    layer = tf.nn.relu(layer)

    return layer

def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    n_features = layer_shape[1:].num_elements()

    layer = tf.reshape(layer, shape=(-1, n_features))

    return layer

def create_fc_layer(input, n_output, activation=None):
    if(len(input.get_shape()) != 2):
        print('input of an fc layer is an fc layer or flatten layer')
        SystemExit(0)

    layer = fully_connected(inputs=input, num_outputs=n_output, activation_fn=activation)
    return layer

def cnn(input):

    with tf.name_scope('conv1'):
        conv1 = create_convolutional_layer(input, 3, 3, 32)

    with tf.name_scope('drop1'):
        drop1 = dropout(inputs=conv1, keep_prob=1, is_training=is_training)

    with tf.name_scope('conv2'):
        conv2 = create_convolutional_layer(drop1, 32, 3, 64)

    with tf.name_scope('drop2'):
        drop2 = dropout(inputs=conv2, keep_prob=1, is_training=is_training)

    with tf.name_scope('conv2'):
        conv3 = create_convolutional_layer(drop2, 64, 3, 64)

    with tf.name_scope('drop3'):
        drop3 = dropout(inputs=conv3, keep_prob=0.5, is_training=is_training)

    with tf.name_scope('flatten'):
        flatten = create_flatten_layer(drop3)

    with tf.name_scope('fc1'):
        fc1 = create_fc_layer(flatten, 1024, activation=tf.nn.relu)

    with tf.name_scope('drop4'):
        drop4 = dropout(inputs=fc1, keep_prob=1, is_training=is_training)

    with tf.name_scope('fc2'):
        fc2 = create_fc_layer(drop4, n_classes)

    return fc2

def fetch_batch(train_images, train_labels, batch_index):
    #print(train_images)
    if(batch_index + batch_size > train_images.shape[0]):
        _x = train_images[batch_index:, :, :, :]
        _y = train_labels[batch_index:, :]
    else:
        _x = train_images[batch_index:batch_index+batch_size, :, :, :]
        _y = train_labels[batch_index:batch_index+batch_size, :]
    return _x, _y

def train(train_images, train_labels, test_images, test_labels):
    n_batches = int(np.ceil(train_images.shape[0] / batch_size))

    X = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(dtype=tf.int64, shape=(None, n_classes))

    logits = cnn(X)

    #y_pred_cls = tf.argmax(logits, 1)
    y_true_cls = tf.argmax(y, 1)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y_true_cls, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope('cross_entropy'):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(xentropy)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        training_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                iteration = epoch * n_batches + batch_index
                X_batch, y_batch = fetch_batch(train_images, train_labels, batch_index)
                # log loss to file
                # summary_str = xentropy_summary.eval(feed_dict={X: X_batch, y: y_batch})
                # filewriter.add_summary(summary_str, iteration)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, is_training: True})

            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, is_training: True})
            acc_test = accuracy.eval(feed_dict={X: test_images, y: test_labels, is_training: False})

            if (epoch % print_every_n_epochs == 0):
                print(epoch, "train_accuracy: ", acc_train, " validation accuracy: ", acc_test)


def main():

    # create class label
    global classes
    tmp_classes = []
    for i in range(n_classes):
        if(i<10):
            tmp_classes.append('0000'+str(i))
        elif(i < 100):
            tmp_classes.append('000'+str(i))
    classes = tmp_classes

    # load_data
    load_data_s_time = time.time()
    dataloader = load_data.Data_Loader(path_to_folder='/home/namntse05438/datasets/GTSRB/GTSRB_train/Final_Training/Images/',
                                        classes=classes,
                                        test_size=0.3)

    train_images, train_labels, test_images, test_labels = dataloader.getTrainTestSets()
    load_data_e_time = time.time()

    print('Load data in %.02fs'%(load_data_e_time-load_data_s_time))

    print(train_images.shape)
    print(test_images.shape)

    train(train_images, train_labels, test_images, test_labels)



main()