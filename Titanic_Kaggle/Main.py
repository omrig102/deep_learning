import tensorflow as tf
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns',1000)

INPUT_SIZE = 5
OUTPUT_SIZE = 2
BATCH_SIZE = 50
LEARNING_RATE = 0.0003
ITERATIONS = 7000

def next_batch(dataset_x,dataset_y=None,batch_size=None,isTrain=True) :
    index = random.randint(0,len(dataset_x) - batch_size)
    batch_x = dataset_x[index:index+batch_size].reshape([batch_size,INPUT_SIZE])
    if isTrain:
        batch_y = dataset_y[index:index+batch_size].reshape([batch_size,OUTPUT_SIZE])
        return batch_x,batch_y

    return batch_x

def one_hot_encode(dataset) :
    results = []
    for i in range(len(dataset)) :
        vector = np.zeros([OUTPUT_SIZE])
        vector[dataset[i]] = 1
        results.append(vector)

    return np.asarray(results)

def read_data(filename,index_col=None,drop_cols=None,category_cols=None) :
    data = pd.read_csv(filename,index_col=index_col)
    if drop_cols is not None :
        for i in range(len(drop_cols)) :
            data = data.drop(drop_cols[i],1)
    if category_cols is not None:
        for i in range(len(category_cols)) :
            data[category_cols[i]] = data[category_cols[i]].astype('category')
            data[category_cols[i]] = data[category_cols[i]].cat.codes

    return data

data = read_data('train.csv',index_col=0,category_cols=['Name','Ticket','Cabin','Embarked','Sex'])
data = data.fillna(0)
data_x = data[data.columns[1:]]
INPUT_SIZE = len(data_x.columns)
data_y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=42)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

x = tf.placeholder(dtype=tf.float32,shape=[None,INPUT_SIZE])
y = tf.placeholder(dtype=tf.float32,shape=[None,OUTPUT_SIZE])

hidden1 = tf.layers.dense(x,units=50,activation=tf.nn.tanh)
hidden2 = tf.layers.dense(hidden1,units=20,activation=tf.nn.tanh)

outputs = tf.layers.dense(hidden2,units=OUTPUT_SIZE,activation=tf.nn.relu)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train = optimizer.minimize(loss)

with tf.Session() as sess :
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(ITERATIONS) :
        batch_x,batch_y = next_batch(X_train,y_train,BATCH_SIZE)
        sess.run(train,feed_dict={x:batch_x,y:batch_y})
        if i % 100 == 0:
            batch_x, batch_y = next_batch(X_test, y_test, len(X_test))
            matches = tf.equal(tf.arg_max(outputs,1),tf.arg_max(y,1))
            acc = tf.reduce_mean(tf.cast(matches,dtype=tf.float32))
            acc_res = sess.run(acc,feed_dict={x:batch_x,y:batch_y})
            print(i,'\tAccuracy : ',acc_res)


    real_results = []
    data = read_data('test.csv',index_col=0,category_cols=['Name','Ticket','Cabin','Embarked','Sex'])
    data = data.fillna(0)
    data = scaler.transform(data)

    batch_x = next_batch(data,batch_size=len(data),isTrain=False)
    results = sess.run(outputs,feed_dict={x:batch_x})
    for i in range(len(results)) :
        real_results.append(np.argmax(results[i]))

    df = pd.DataFrame(real_results)
    df.to_csv('results.csv')