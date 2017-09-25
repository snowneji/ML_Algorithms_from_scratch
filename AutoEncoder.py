from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib

#First MNIST data
mnist = input_data.read_data_sets('/tmp/data/',one_hot=True)



# Some hyper-parameters:
learning_rate = 0.01
n_steps = 50000
batch_size = 256

display_step = 1000
examles_to_show = 10


# Architecture pars:
n_input = 28*28
n_h1 = 256
n_h2 = 128



# Input:
X = tf.placeholder("float",[None,n_input])

# 2 layer Encoder:
w1 = tf.Variable(tf.random_normal([n_input,n_h1]))
b1 = tf.Variable(tf.random_normal([n_h1]))
h1 = tf.add(tf.matmul(X,w1),b1)
l1 = tf.nn.sigmoid(h1)

w2 = tf.Variable(tf.random_normal([n_h1,n_h2]))
b2 = tf.Variable(tf.random_normal([n_h2]))
h2 = tf.add(tf.matmul(l1,w2),b2)
l2 = tf.nn.sigmoid(h2)


# 2 layer decoder:

de_w1 = tf.Variable(tf.random_normal([n_h2,n_h1]))
de_b1 = tf.Variable(tf.random_normal([n_h1]))
de_h1 = tf.add(tf.matmul(l2,de_w1),de_b1)
de_l1 = tf.nn.sigmoid(de_h1)

de_w2 = tf.Variable(tf.random_normal([n_h1,n_input]))
de_b2 = tf.Variable(tf.random_normal([n_input]))
de_h2 = tf.add(tf.matmul(de_l1,de_w2),de_b2)
de_l2=  tf.nn.sigmoid(de_h2)


# preds
y_pred = de_l2
y_true = X # Input as the label for auto-encoder

#Loss and Optimizer:
loss = tf.reduce_mean(tf.pow(y_true-y_pred,2)) #mean squared error
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)



#Initialize and training:

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	#Train:
	for i in range(1,n_steps+1):
		batch_x, _ = mnist.train.next_batch(batch_size)

		_, l = sess.run([optimizer, loss],feed_dict={X:batch_x})
		#every n step display:
		if i%display_step==0:
			print("Step: %d -- Loss: %.3f"%(i,l))


	# Test:
	# Run the images in the through the Encoder to see how 
	# they are reconstructed

	n = 4
	# matrix containing 16 digits: 28*4 by 28*4

	orig = np.empty((28*n,28*n))
	recn = np.empty((28*n,28*n))

	for i in range(n):
		batch_x, _ = mnist.test.next_batch(n)
		res = sess.run(de_l2, feed_dict = {X: batch_x})

		for j in range(n):
			# Original Image:
			orig[i*28:(i+1)*28, j*28:(j+1)*28] = batch_x[j].reshape((28,28))

		# New Image:
		for k in range(n):
			recn[i*28:(i+1)*28, k*28:(k+1)*28] = res[k].reshape([28,28])


	print('Original Image:')
	plt.figure(figsize=(n,n))
	plt.imshow(orig,origin='upper',cmap='gray')
	matplotlib.image.imsave("/home/snowneji/Desktop/orig.png",orig)
	plt.show()

	print('Autoencoder Produced:')
	plt.figure(figsize=(n,n))
	plt.imshow(recn,origin='upper',cmap='gray')
	matplotlib.image.imsave("/home/snowneji/Desktop/recn.png",recn)
	plt.show()



