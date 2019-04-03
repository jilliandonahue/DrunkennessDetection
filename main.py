# Speech Emotion and Drunkenness Detection
# Written by: Josh Miller, Jill Donahue, and Ben Schmitz
# Adapted from code by: Somayeh (Bahar) Shahsavarani

import scipy.io.wavfile
import scipy.signal
import skimage.transform as skit
import numpy as np
import tensorflow as tf
import os


# 1 neutral 2 happy 3 sad 4 angry 5 drunk
trainDir1 = "/home/jdonahu2/Final Project/trainneutral/"
trainDir2 = "/home/jdonahu2/Final Project/trainhappy/"
trainDir3 = "/home/jdonahu2/Final Project/trainsad/"
trainDir4 = "/home/jdonahu2/Final Project/trainangry/"
trainDir5 = "/home/jdonahu2/Final Project/traindrunk/"
testDir1 = "/home/jdonahu2/Final Project/testneutral/"
testDir2 = "/home/jdonahu2/Final Project/testhappy/"
testDir3 = "/home/jdonahu2/Final Project/testsad/"
testDir4 = "/home/jdonahu2/Final Project/testangry/"
testDir5 = "/home/jdonahu2/Final Project/testdrunk/"


s_train = []  # empty matrices for holding all of spectrogram data
s_test = []
train_x = []
test_x = []
train_y = []
test_y = []

# number of classes depending on the number of target emotions in the database
n_classes = 5

# batch size depending on the number of training data you have
batch_size = 64

# number of training epochs
hm_epochs = 100

# height of the image
hm_pixels1 = 129

# width of the image
hm_pixels2 = 129

counter = 0 # count the number of files in each directory to properly assign labels
global_count = 0

for file in os.listdir(trainDir1): # for each file in train neutral file directory

    print(trainDir1+file)
    fs,x = scipy.io.wavfile.read(trainDir1+file)
    noise_amp = np.sqrt(np.mean(x ** 2)) / (10**(15/10))
    noise = noise_amp * (-1 + (2* np.random.rand(len(x)*2)))
    x = np.concatenate((x,noise))
    
    _, _, Sxx = scipy.signal.spectrogram(x, fs, 'hamming', 80, 70, 512)
    Sxx = 20 * np.log10(Sxx[0:129, 0:])
    Sxx = skit.resize(Sxx, (129, 129), 1, anti_aliasing=False)

    s_train = (Sxx)  # append each new spectrogram to training data
    print("train time")
    current_class = [1,0,0,0,0]
    s_train = np.array(s_train)
    train_x.append(np.reshape(s_train, (hm_pixels1 * hm_pixels2)))  # append current file spectrogram to training data
    counter = counter + 1
    global_count = global_count + 1

 # train_y is the label matrix for training
train_y = train_y + (counter * [current_class]) # drunk they're all drunk
counter = 0 # reset file counter

for file in os.listdir(trainDir2):  # for each file in train happy file directory

    print(trainDir2 + file)
    fs, x = scipy.io.wavfile.read(trainDir2 + file)

    noise_amp = np.sqrt(np.mean(x ** 2)) / (10 ** (15 / 10))
    noise = noise_amp * (-1 + (2 * np.random.rand(len(x) * 2)))
    x = np.concatenate((x, noise))

    _, _, Sxx = scipy.signal.spectrogram(x, fs, 'hamming', 80, 70, 512)
    Sxx = 20 * np.log10(Sxx[0:129, 0:])
    Sxx = skit.resize(Sxx, (129, 129), 1, anti_aliasing=False)

    s_train = (Sxx)  # append each new spectrogram to training data
    print("train time")
    current_class = [0, 1, 0, 0, 0]
    s_train = np.array(s_train)
    train_x.append(np.reshape(s_train, (hm_pixels1 * hm_pixels2)))  # append current file spectrogram to training data
    counter = counter + 1
    global_count = global_count + 1


 # train_y is the label matrix for training
train_y = train_y + (counter * [current_class]) # drunk they're all drunk
counter = 0  # reset file counter

for file in os.listdir(trainDir3):  # for each file in train sad file directory

    print(trainDir3 + file)
    fs, x = scipy.io.wavfile.read(trainDir3 + file)

    noise_amp = np.sqrt(np.mean(x ** 2)) / (10 ** (15 / 10))
    noise = noise_amp * (-1 + (2 * np.random.rand(len(x) * 2)))
    x = np.concatenate((x, noise))

    _, _, Sxx = scipy.signal.spectrogram(x, fs, 'hamming', 80, 70, 512)
    Sxx = 20 * np.log10(Sxx[0:129, 0:])
    Sxx = skit.resize(Sxx, (129, 129), 1, anti_aliasing=False)

    s_train = (Sxx)  # append each new spectrogram to training data
    print("train time")
    current_class = [0, 0, 1, 0, 0]
    s_train = np.array(s_train)
    train_x.append(np.reshape(s_train, (hm_pixels1 * hm_pixels2)))  # append current file spectrogram to training data
    counter = counter + 1
    global_count = global_count + 1


 # train_y is the label matrix for training
train_y = train_y + (counter * [current_class]) # drunk they're all drunk
counter = 0  # reset file counter

for file in os.listdir(trainDir4):  # for each file in train angry file directory

    print(trainDir4 + file)
    fs, x = scipy.io.wavfile.read(trainDir4 + file)

    noise_amp = np.sqrt(np.mean(x ** 2)) / (10 ** (15 / 10))
    noise = noise_amp * (-1 + (2 * np.random.rand(len(x) * 2)))
    x = np.concatenate((x, noise))

    _, _, Sxx = scipy.signal.spectrogram(x, fs, 'hamming', 80, 70, 512)
    Sxx = 20 * np.log10(Sxx[0:129, 0:])
    Sxx = skit.resize(Sxx, (129, 129), 1, anti_aliasing=False)

    s_train = (Sxx)  # append each new spectrogram to training data
    print("train time")
    current_class = [0, 0, 0, 1, 0]
    s_train = np.array(s_train)
    train_x.append(np.reshape(s_train, (hm_pixels1 * hm_pixels2)))  # append current file spectrogram to training data
    counter = counter + 1
    global_count = global_count + 1

 # train_y is the label matrix for training
train_y = train_y + (counter * [current_class]) # drunk they're all drunk
counter = 0  # reset file counter

for file in os.listdir(trainDir5):  # for each file in train drunk file directory

    print(trainDir5 + file)
    fs, x = scipy.io.wavfile.read(trainDir5 + file)

    noise_amp = np.sqrt(np.mean(x ** 2)) / (10 ** (15 / 10))
    noise = noise_amp * (-1 + (2 * np.random.rand(len(x) * 2)))
    x = np.concatenate((x, noise))

    _, _, Sxx = scipy.signal.spectrogram(x, fs, 'hamming', 80, 70, 512)
    Sxx = 20 * np.log10(Sxx[0:129, 0:])
    Sxx = skit.resize(Sxx, (129, 129), 1, anti_aliasing=False)

    s_train = (Sxx)  # append each new spectrogram to training data
    print("train time")
    current_class = [0, 0, 0, 0, 1]
    s_train = np.array(s_train)
    train_x.append(np.reshape(s_train, (hm_pixels1 * hm_pixels2)))  # append current file spectrogram to training data
    counter = counter + 1
    global_count = global_count + 1

 # train_y is the label matrix for training
train_y = train_y + (counter * [current_class]) # drunk they're all drunk
counter = 0  # reset file counter

###########################################
         # READ TEST DATA NOW
###########################################


for file in os.listdir(testDir1):  # for each file in test neutral file directory

    print(testDir1 + file)
    fs, x = scipy.io.wavfile.read(testDir1 + file)

    _, _, Sxx = scipy.signal.spectrogram(x, fs, 'hamming', 80, 70, 512)
    Sxx = 20 * np.log10(Sxx[0:129, 0:])
    Sxx = skit.resize(Sxx, (129, 129), 1, anti_aliasing=False)

    s_test = (Sxx)  # append each new spectrogram to training data
    print("test time")
    current_class = [1, 0, 0, 0, 0]
    s_test = np.array(s_test)
    test_x.append(np.reshape(s_test, (hm_pixels1 * hm_pixels2)))  # append current file spectrogram to training data
    counter = counter + 1

 # train_y is the label matrix for training
test_y = test_y + (counter * [current_class]) # drunk they're all drunk
counter = 0  # reset file counter

for file in os.listdir(testDir2):  # for each file in test happy file directory

    print(testDir2 + file)
    fs, x = scipy.io.wavfile.read(testDir2 + file)

    _, _, Sxx = scipy.signal.spectrogram(x, fs, 'hamming', 80, 70, 512)
    Sxx = 20 * np.log10(Sxx[0:129, 0:])
    Sxx = skit.resize(Sxx, (129, 129), 1, anti_aliasing=False)

    s_test = (Sxx)  # append each new spectrogram to training data
    print("test time")
    current_class = [0, 1, 0, 0, 0]
    s_test = np.array(s_test)
    test_x.append(np.reshape(s_test, (hm_pixels1 * hm_pixels2)))  # append current file spectrogram to training data
    counter = counter + 1

 # train_y is the label matrix for training
test_y = test_y + (counter * [current_class]) # drunk they're all drunk
counter = 0  # reset file counter

for file in os.listdir(testDir3):  # for each file in test sad file directory

    print(testDir3 + file)
    fs, x = scipy.io.wavfile.read(testDir3 + file)

    _, _, Sxx = scipy.signal.spectrogram(x, fs, 'hamming', 80, 70, 512)
    Sxx = 20 * np.log10(Sxx[0:129, 0:])
    Sxx = skit.resize(Sxx, (129, 129), 1, anti_aliasing=False)

    s_test = (Sxx)  # append each new spectrogram to training data
    print("test time")
    current_class = [0, 0, 1, 0, 0]
    s_test = np.array(s_test)
    test_x.append(np.reshape(s_test, (hm_pixels1 * hm_pixels2)))  # append current file spectrogram to training data
    counter = counter + 1

 # train_y is the label matrix for training
test_y = test_y + (counter * [current_class]) # drunk they're all drunk
counter = 0  # reset file counter

for file in os.listdir(testDir4):  # for each file in test angry file directory

    print(testDir4 + file)
    fs, x = scipy.io.wavfile.read(testDir4 + file)

    _, _, Sxx = scipy.signal.spectrogram(x, fs, 'hamming', 80, 70, 512)
    Sxx = 20 * np.log10(Sxx[0:129, 0:])
    Sxx = skit.resize(Sxx, (129, 129), 1, anti_aliasing=False)

    s_test = (Sxx)  # append each new spectrogram to training data
    print("test time")
    current_class = [0, 0, 0, 1, 0]
    s_test = np.array(s_test)
    test_x.append(np.reshape(s_test, (hm_pixels1 * hm_pixels2)))  # append current file spectrogram to training data
    counter = counter + 1

 # train_y is the label matrix for training
test_y = test_y + (counter * [current_class]) # drunk they're all drunk
counter = 0  # reset file counter

for file in os.listdir(testDir5):  # for each file in test drunk file directory

    print(testDir5 + file)
    fs, x = scipy.io.wavfile.read(testDir5 + file)

    _, _, Sxx = scipy.signal.spectrogram(x, fs, 'hamming', 80, 70, 512)
    Sxx = 20 * np.log10(Sxx[0:129, 0:])
    Sxx = skit.resize(Sxx, (129, 129), 1, anti_aliasing=False)

    s_test = (Sxx)  # append each new spectrogram to training data
    print("test time")
    current_class = [0, 0, 0, 0, 1]
    s_test = np.array(s_test)
    test_x.append(np.reshape(s_test, (hm_pixels1 * hm_pixels2)))  # append current file spectrogram to training data
    counter = counter + 1

# train_y is the label matrix for training
test_y = test_y + (counter * [current_class]) # drunk they're all drunk


# define a placeholder for the feature matrices
x = tf.placeholder('float', [None, hm_pixels1 * hm_pixels2])

# define a placeholder for the label matrices
y = tf.placeholder('float')

# the dropout probability
keep_prob = tf.placeholder(tf.float32)

# todo: reshape train_x and train_y so they are 2D not 3D

# define convolutional layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# define pooling layer
def avgpool2d(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([10, 10, 1, 8])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 8, 16])),
               'W_fc': tf.Variable(tf.random_normal([33 * 33 * 16, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([8])),
              'b_conv2': tf.Variable(tf.random_normal([16])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, hm_pixels1, hm_pixels2, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = avgpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = avgpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 33 * 33 * 16])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc_drop = tf.nn.dropout(fc, keep_prob)

    output = tf.matmul(fc_drop, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        for epoch in range(hm_epochs):
            epoch_loss = 0

            # build batches
            i = 0
            while i < global_count: # total amount of files written in
                start = i
                end = i + batch_size
                
                if end > len(train_x):
                    i = global_count+1
                    
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])


                # print(i)

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

                epoch_loss += c
                i += batch_size
                training_accuracy = accuracy.eval({x: train_x, y: train_y, keep_prob: 1})
                test_accuracy = accuracy.eval({x: test_x, y: test_y, keep_prob: 1})
                # make confusion matrix by hand
                loss_test = sess.run(cost, feed_dict={x: test_x, y: test_y, keep_prob: 1})
                print("Train Accuracy: ", training_accuracy)
                print("Test Accuracy: ", test_accuracy)
                print("Training Loss: ", c)
                print("Testing Loss: ", loss_test)
                
            label = tf.argmax(y, axis=1)
            predict = tf.argmax(prediction, axis=1)
            results = tf.confusion_matrix(label,predict, num_classes=n_classes)
            
            print("Confusion Matrix: ")
            print(results.eval({x:test_x, y:test_y, keep_prob:1}))

            print("Epoch #: ", epoch+1)
            print("Total loss for one epoch: ", epoch_loss)
            
train_neural_network(x)

