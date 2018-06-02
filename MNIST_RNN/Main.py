import tensorflow as tf
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/mnist',one_hot=True)

INPUT_SIZE = 28
OUTPUT_SIZE = 10
NUM_STEPS = 28
BATCH_SIZE = 50
LEARNING_RATE = 0.0003
ITERATIONS = 2000

x = tf.placeholder(dtype=tf.float32,shape=[None,NUM_STEPS * INPUT_SIZE])
y = tf.placeholder(dtype=tf.float32,shape=[None,1,OUTPUT_SIZE])

reshape = tf.reshape(x,shape=[-1,NUM_STEPS,INPUT_SIZE])

cell = rnn.GRUCell(num_units=50,activation=tf.nn.relu)

cell = rnn.OutputProjectionWrapper(cell,OUTPUT_SIZE)
outputs,states = tf.nn.dynamic_rnn(cell,reshape,dtype=tf.float32)

outputs = outputs[:,-1,:]

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(ITERATIONS) :
        batch_x,batch_y = mnist.train.next_batch(BATCH_SIZE)
        batch_y = batch_y.reshape([-1,1,OUTPUT_SIZE])
        sess.run(train,feed_dict={x:batch_x,y:batch_y})
        if i % 100 == 0:
            batch_x,batch_y = mnist.test.next_batch(2000)
            batch_y = batch_y.reshape([-1, 1, OUTPUT_SIZE])

            matches = tf.equal(tf.argmax(outputs,1),tf.argmax(tf.reshape(y,shape=[-1,OUTPUT_SIZE]),1))
            acc = tf.reduce_mean(tf.cast(matches,dtype=tf.float32))
            acc_res = sess.run(acc,feed_dict={x:batch_x,y:batch_y})
            print(i,'\tAccuracy : ',acc_res)
