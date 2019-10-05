import tensorflow as tf
import numpy as np
import cv2
from read_write_tfrecords import *


shape_name = ['one','ellipse','etc']
char_name = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','default']

size = 48
n_shuffle = 100000
batch_size = 50
# n_class = 3
n_class = 37
learning_rate = 0.001

dataset = read_tfrecords(['./tfrecords/'+str+'.tfrecords' for str in char_name])

dataset = dataset.repeat()
dataset = dataset.shuffle(n_shuffle)
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


X = tf.placeholder(np.float32,shape=[None,size,size,1],name="x")
Y = tf.placeholder(np.float32,shape=[None,n_class],name="y")
keep_prob = tf.placeholder(np.float32)

conv1 = tf.layers.conv2d(inputs=X,filters=32,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu,name="conv1")
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=[2,2],name="pool1")
conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu,name="conv2")
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=[2,2],name="pool2")
conv3 = tf.layers.conv2d(inputs=pool2,filters=128,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu,name="conv3")
pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=[2,2],name="pool3")

pool_flat = tf.reshape(pool3,[-1,size*size*2],name="pool_flat")

fc1 = tf.layers.dense(inputs=pool_flat,units=1024,activation=tf.nn.relu,name="fc1")
fc1_drop = tf.layers.dropout(fc1,rate=keep_prob,name="fc1_drop")
fc2 = tf.layers.dense(inputs=fc1_drop,units=256,activation=tf.nn.relu,name="fc2")
fc2_drop = tf.layers.dropout(fc2,rate=keep_prob,name="fc2_drop")
logits = tf.layers.dense(inputs=fc2_drop,units=n_class,name="logits")


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(Y,logits))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    saver = tf.train.Saver()

    for i in range(50000):
        image_batch,label_batch = sess.run(next_element)
        label_batch = tf.one_hot(label_batch,depth=n_class).eval(session=sess)
        label_batch = np.asarray(label_batch, dtype=np.float32)
        sess.run(optimizer,feed_dict={X:image_batch,Y:label_batch,keep_prob:0.8})
        if (i + 1) % 100 == 0:
            co = sess.run(cost,feed_dict={X:image_batch,Y:label_batch,keep_prob:1.0})
            print("{0} : {1}".format(i + 1,co))
            saver.save(sess,'saver/char/char',global_step=i+1)

