import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/mnist',one_hot=True)

INPUT_SIZE = 784
OUTPUT_SIZE = 10
BATCH_SIZE = 50
LEARNING_RATE = 0.0003
ITERATIONS = 1000

x = tf.placeholder(dtype=tf.float32,shape=[None,INPUT_SIZE])
y = tf.placeholder(dtype=tf.float32,shape=[None,OUTPUT_SIZE])

reshape = tf.reshape(x,shape=[-1,int(math.sqrt(INPUT_SIZE)),int(math.sqrt(INPUT_SIZE)),1])

conv1 = tf.layers.conv2d(reshape,filters=50,kernel_size=6,strides=1,activation=tf.nn.tanh,padding='SAME')
pooling1 = tf.layers.max_pooling2d(conv1,pool_size=2,strides=2)

conv2 = tf.layers.conv2d(pooling1,filters=20,kernel_size=6,strides=1,activation=tf.nn.tanh,padding='SAME')
pooling2 = tf.layers.max_pooling2d(conv2,pool_size=2,strides=2)

conv3 = tf.layers.conv2d(pooling2,filters=10,kernel_size=6,strides=1,activation=tf.nn.tanh,padding='SAME')
pooling3 = tf.layers.max_pooling2d(conv3,pool_size=2,strides=2)


flatten_layer = tf.layers.flatten(pooling3)
dense1 = tf.layers.dense(flatten_layer,units=100,activation=tf.nn.tanh)

outputs = tf.layers.dense(dense1,units=OUTPUT_SIZE,activation=tf.nn.relu)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(ITERATIONS) :
        batch_x,batch_y = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train,feed_dict={x:batch_x,y:batch_y})
        if i % 100 == 0:
            batch_x,batch_y = mnist.test.next_batch(2000)
            matches = tf.equal(tf.argmax(outputs,1),tf.argmax(y,1))
            acc = tf.reduce_mean(tf.cast(matches,dtype=tf.float32))
            acc_res = sess.run(acc,feed_dict={x:batch_x,y:batch_y})
            print(i,'\tAccuracy : ',acc_res)
