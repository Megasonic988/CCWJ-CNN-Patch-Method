import tensorflow as tf
import numpy as np

# Extract training and test data
images_filenames = ['images/BM50xajpg_subimages.npy', 'images/haz50xajpg_subimages.npy']
labels_filenames = ['images/BM50xajpg_labels.npy', 'images/haz50xajpg_labels.npy']
images = []
labels = []
for filename in images_filenames:
    images += list(map(lambda i: i.ravel(), np.load(filename) / 255))
for filename in labels_filenames:
    labels_for_image = list(np.load(filename))
    labels_for_image = list(map(lambda l: [1, 0] if l == 'PF - Primary Ferrite' else [0, 1], labels_for_image))
    labels += labels_for_image

train_images = images[0:1300] + images[2500:3863]
test_images = images[1300:2500]
train_labels = labels[0:1300] + labels[2500:3863]
test_labels = labels[1300:2500]

def batch(data, labels, batch_size):
    data_batch = []
    labels_batch = []
    for _ in range(batch_size):
        rand = np.random.randint(0, len(data))
        data_batch.append(data[rand])
        labels_batch.append(labels[rand])
    return [data_batch, labels_batch]

# CNN model
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 2500])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
conv_W1 = weight_variable([5, 5, 1, 16])
b1 = bias_variable([16])
conv_W2 = weight_variable([5, 5, 16, 32])
b2 = bias_variable([32])
fc_W3 = weight_variable([13 * 13 * 32, 128])
b3 = bias_variable([128])
fc_W4 = weight_variable([128, 2])
b4 = bias_variable([2])

reshape = tf.reshape(x, [-1, 50, 50, 1])
conv1 = tf.nn.bias_add(conv2d(reshape, conv_W1), b1)
conv1 = tf.nn.relu(conv1)
pool1 = pool(conv1)
conv2 = tf.nn.bias_add(conv2d(pool1, conv_W2), b2)
conv2 = tf.nn.relu(conv2)
pool2 = pool(conv2)
fc1 = tf.nn.relu(tf.matmul(tf.reshape(pool2, [-1, 13 * 13 * 32]), fc_W3) + b3)
dropout = tf.nn.dropout(fc1, keep_prob)
fc2 = tf.nn.relu(tf.matmul(dropout, fc_W4) + b4)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc2))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch_xs, batch_ys = batch(train_images, train_labels, 10)
        reshape_val, conv1_val, pool1_val, conv2_val, pool2_val, fc1_val, dropout_val, fc2_val, loss_val, train_step_val = sess.run([
            reshape,
            conv1,
            pool1,
            conv2,
            pool2,
            fc1,
            dropout,
            fc2,
            loss,
            train_step
        ], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        print(loss_val)
    pred = tf.nn.softmax(fc2)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: test_images, y: test_labels, keep_prob: 1.0}))