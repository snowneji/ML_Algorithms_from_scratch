import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.misc import imshow

def OHE(label,nclass):
	label_dat = np.zeros((len(label),nclass))
	for i in range(len(label)):
		label_dat[i,label[i]] = 1
	return label_dat




if __name__ =='__main__':
	# Load data:
	train = pd.read_csv('/home/snowneji/MNIST/train.csv')
	test = pd.read_csv('/home/snowneji/MNIST/test.csv')


	IMAGE_SIZE = int(np.sqrt(test.shape[1])) # Image size
	SPLIT_SIZE = 0.2  # Train Vali Ratio
	N_CLASS = 10
	LEARNING_RATE = 0.001
	TRAINING_ITER = 5000
	BATCH_SIZE = 50
	VALIDATION_SIZE = 2000


	#get arrays:
	label = train['label'].values
	label = OHE(label,N_CLASS) # OHE label data
	train_mat = train.ix[:,1:].values
	test_mat = train.ix[:,1:].values


	#visualize a random image:
	# ind = np.random.randint(0,len(train_mat))
	# rand_img = train_mat[ind,:].reshape(IMAGE_SIZE,IMAGE_SIZE)
	# imshow(rand_img)
	# print(label[ind])


	#normalize the data:
	train_mat = train_mat/255.0
	test_mat = test_mat/255.0



	#split train into train and valid data:

	# Shuffling the data
	permu = np.random.permutation(len(train))
	train_mat = train_mat[permu,:]
	label = label[permu,:]

	# split:
	vali_ind = int(len(train_mat)*0.2)
	vali_dat = train_mat[:vali_ind]
	train_dat= train_mat[vali_ind:]
	vali_label = label[:vali_ind,:]
	train_label= label[vali_ind:,:]


	






	#################
	#################
	#################
	# Model Graph now:
	#################
	#################
	#################
	#Input:
	x = tf.placeholder(tf.float32,shape=[None,IMAGE_SIZE*IMAGE_SIZE])
	y_ = tf.placeholder(tf.float32,shape=[None,N_CLASS])


	# Layer 1:   1 -> 32 feature map

	#image_size: 28
	image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,1 ])
	W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.01))  #weight
	b_conv1 = tf.Variable(tf.constant(0.01,shape=[32])) #bias
	conv1 = tf.nn.conv2d(image,W_conv1,strides=[1,1,1,1],padding='SAME') #Convolution
	relu1 = tf.nn.relu(conv1+b_conv1) # relu transform
	pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1],strides = [1,2,2,1],padding='SAME') #image_size:14


	# Layer 2: 32 -> 64 feature map
	#image_size:14
	W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.01))  #weight
	b_conv2 = tf.Variable(tf.constant(0.01,shape=[64])) #bias
	conv2 = tf.nn.conv2d(pool1,W_conv2,strides=[1,1,1,1],padding='SAME') #Convolution
	relu2 = tf.nn.relu(conv2+b_conv2) # relu transform
	#image_size: 7
	pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1],strides = [1,2,2,1],padding='SAME')



	# Layer 3: 64 -> 128 feature map
	W_conv3 = tf.Variable(tf.truncated_normal([5,5,64,128],stddev=0.01))  #weight
	b_conv3 = tf.Variable(tf.constant(0.01,shape=[128])) #bias
	conv3 = tf.nn.conv2d(pool2,W_conv3,strides=[1,1,1,1],padding='SAME') #Convolution
	relu2 = tf.nn.relu(conv3+b_conv3) # relu transform










	#Dense Layer 1:
	# each image now become [7,7,128]
	# to dense it: it now because 7*7*128
	# let's set: 7*7*128  -> 1024 for dense layer

	
	flat_pool2 = tf.reshape(relu2,[-1,7*7*128])#Flatten

	W_fc1 = tf.Variable(tf.truncated_normal([7*7*128,1024],stddev=0.01)) 
	b_fc1 = tf.Variable(tf.constant(0.01,shape=[1024])) #bias


	fc1 = tf.matmul(flat_pool2,W_fc1)+b_fc1
	relu_full1 = tf.nn.relu(fc1)
	# Add some fancy dropout to prevent overfitting:
	keep_prob = tf.placeholder(tf.float32) # dropout rate as a variable
	drop1 = tf.nn.dropout(relu_full1,keep_prob)





	#Dense Layer 2:

	# let's set:  1024 -> 512
	

	W_fc2 = tf.Variable(tf.truncated_normal([1024,512],stddev=0.01)) 
	b_fc2 = tf.Variable(tf.constant(0.01,shape=[512])) #bias


	fc2 = tf.matmul(drop1,W_fc2)+b_fc2
	relu_full2 = tf.nn.relu(fc2)
	# Add some fancy dropout to prevent overfitting:
	
	drop2 = tf.nn.dropout(relu_full2,keep_prob)














	## Softmax to get result:
	# 512 -> 10
	W_fc3 = tf.Variable(tf.truncated_normal([512,N_CLASS],stddev=0.01)) 
	b_fc3 = tf.Variable(tf.constant(0.01,shape=[N_CLASS])) #bias
	softmax_res = tf.matmul(drop2,W_fc3)+b_fc3
	y = softmax_res#tf.nn.softmax(softmax_res)



	## Add Optimization algo
	cross_entr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)) #-tf.reduce_sum(y_*tf.log(y)) #Cost
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entr)


	## Evaluation:
	print(y)
	print(y_)
	pred = tf.equal(tf.argmax(y,1),  tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(pred,tf.float32))


	#prediction:
	predict = tf.argmax(y,1)







	#next_batch function variables setup:
	epochs_completed = 0
	index_in_epoch = 0
	num_examples = train_dat.shape[0]

	# Next Batch Helper Function:
	def next_batch(batch_size):
		global train_dat
		global train_label
		global index_in_epoch
		global epochs_completed

		start = index_in_epoch
		index_in_epoch += batch_size

		# if index exceed the length of data, shuffle the data and continue:
		if index_in_epoch > num_examples:
			epochs_completed += 1
			# permu = np.random.permutation(num_examples)
			# train_dat = train_dat[permu,:]
			# train_label = train_label[permu,:]
			# Begin next epoch:
			start = 0 # reset start counter
			index_in_epoch = 0 # reset index counter
			index_in_epoch += batch_size # update
			assert batch_size <= num_examples #check point

		end = index_in_epoch
		return(train_dat[start:end,:], train_label[start:end,:])





	#Training:
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())



	#Visualize:
	train_acc = []
	vali_acc = []
	x_range = []
	display_step = 1



	for i in range(TRAINING_ITER):

		batch_x,batch_y = next_batch(BATCH_SIZE)


		if i%100 == 0 or (i+1)==TRAINING_ITER:

			tr_ac = accuracy.eval(feed_dict = {
				x:batch_x, 
				y_:batch_y,
				keep_prob:1.0})


			if(VALIDATION_SIZE):
				vl_ac = accuracy.eval(feed_dict={
					x:vali_dat[0:BATCH_SIZE], 
					y_: vali_label[0:BATCH_SIZE], 
					keep_prob:1.0})




				print("training accuracy /  vali accuracy =>  %.2f / %.2f for step %d" %(tr_ac,vl_ac,i))
				vali_acc.append(vl_ac)


			train_acc.append(tr_ac)
			x_range.append(i)



		#TRAIN:
		sess.run(train_step, feed_dict={x: batch_x, y_:batch_y, keep_prob: 0.9})



### TEST AND SUBMISSION ON KAGGLE:


test = test.astype(np.float32)
test = test/255.0

#initialize:
predicted_label = np.zeros(test.shape[0])

for i in range(0,test.shape[0]/BATCH_SIZE):
	predicted_label[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = predict.eval(feed_dict=
		{
		x:test[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
		keep_prob : 1.0
		})


np.savetxt(
	'/home/snowneji/Desktop/yf_sub.csv',
	np.c_[range(1,len(test)+1),predicted_label],
	delimiter = ',',
	header = "ImageId,Label",
	comments = '',
	fmt = '%d'
	)








