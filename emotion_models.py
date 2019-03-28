import tensorflow as tf
import numpy as np 
from utils_emotion import *
import time
import sys

def main_training():

	data_path = '../fer2013.csv'
	images, emotions = load_fer2013(data_path)
	num_classes = len(emotions[0])
	emotion_labels = find_labels()
	batch_generator = gen_batch_function(images, emotions)
	load = True
	train = True
	network = 'CNN'

	## Paramater
	LEARNING_RATE = 1e-4
	BATCH_SIZE = 32
	EPOCHS = 10
	num_examples = len(images)

	## Define Network 
	if network == 'simple':
		network = BasicNN(num_classes=num_classes, learning_rate=LEARNING_RATE)
	elif network =='CNN':
		network = BasicCNN(num_classes=num_classes, learning_rate=LEARNING_RATE)

	if load == True:
		network.load_model()
		print("Network loaded...")
	if train == True:
		print("Training network...")
		network.train(images, emotions, batch_size=BATCH_SIZE, epochs=EPOCHS, num_examples=num_examples, batch_generator=batch_generator)


####################### Network Classes ##################################


###################### Basic CNN ########################################
class BasicCNN:
	def __init__(self, num_classes, learning_rate):
		self.save_file = './emotion_detection/model.ckpt'
		self.save_epoch = 2
		self.learning_rate=learning_rate
		self.num_classes = num_classes
		self.input_shape = (48, 48)
		self.x = tf.placeholder(tf.float32, [None, 48, 48, 1])
		self.y = tf.placeholder(tf.float32, [None, self.num_classes])
		self.sess = tf.InteractiveSession()

		layer1 = tf.layers.conv2d(self.x, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
		layer1 = tf.layers.max_pooling2d(layer1, pool_size=2, strides=2)

		layer2 = tf.layers.conv2d(layer1, filters=64, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
		layer2 = tf.layers.max_pooling2d(layer2, pool_size=2, strides=2)

		layer3 = tf.layers.conv2d(layer2, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu)
		layer3 = tf.layers.max_pooling2d(layer3, pool_size=2, strides=2)

		dense1 = tf.contrib.layers.flatten(layer3)
		dense1 = tf.layers.dense(dense1, units=256, activation=tf.nn.relu)
		self.out = tf.layers.dense(dense1, units=num_classes)

		predictions = {
						# Generate predictions (for PREDICT and EVAL mode)
					    "classes": tf.argmax(input=self.out, axis=1),
					    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the`logging_hook`.
					    "probabilities": tf.nn.softmax(self.out, name="softmax_tensor")}

		self.loss = tf.losses.softmax_cross_entropy(logits=self.out, onehot_labels=self.y, )
		self.error = tf.reduce_mean(self.loss)

		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = optimizer.minimize(self.error)

		self.sess.run(tf.global_variables_initializer())

	def train(self, images, emotions, batch_size, epochs, num_examples, batch_generator):

		for epoch in range(epochs):
			start_time = time.time()
			total_loss = 0
			batch_num = 0
			for img, lab in batch_generator(batch_size):

				_, los, err = self.sess.run([self.train_op, self.loss, self.error], feed_dict = {self.x: img,
																								self.y: lab})
				total_loss += err
				batch_num += 1

			print("Epoch {0} Loss: {1:1f} Time: {2}".format(epoch+1, total_loss/batch_num, time.time()-start_time))

			# Save after every 10 epochs
			if epoch % self.save_epoch == 0:
				self.save_model()


	def predict(self, image):
		pred = tf.nn.softmax(self.out)
		pred_emotion = tf.argmax(pred, 1)
		prediction, emot = self.sess.run([pred, pred_emotion], feed_dict={self.x: image}) 
																							# self.y: emotion})
		return prediction, emot

	def save_model(self):
		saver = tf.train.Saver()
		saver.save(self.sess, self.save_file)

	def load_model(self):
		saver = tf.train.Saver()
		saver.restore(self.sess, tf.train.latest_checkpoint('./emotion_detection'))


###################### Basic NN ########################################


class BasicNN:
	def __init__(self, num_classes, learning_rate):
		self.num_classes = num_classes
		self.input_shape = 2304
		self.hidden_layer = 200
		self.learning_rate = learning_rate
		self.save_file = './emotion_detection/simple_nn/model.ckpt'
		self.save_epoch = 2

		self.x = tf.placeholder(tf.float32, [None, self.input_shape])
		self.y = tf.placeholder(tf.float32, [None, self.num_classes])
		self.sess = tf.InteractiveSession()

		layer1 = tf.layers.dense(self.x, units=self.hidden_layer, activation=tf.nn.relu)
		self.out = tf.layers.dense(layer1, units=num_classes)

		predictions = {
						# Generate predictions (for PREDICT and EVAL mode)
					    "classes": tf.argmax(input=self.out, axis=1),
					    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the`logging_hook`.
					    "probabilities": tf.nn.softmax(self.out, name="softmax_tensor")}

		self.loss = tf.losses.softmax_cross_entropy(logits=self.out, onehot_labels=self.y, )
		self.error = tf.reduce_mean(self.loss)

		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = optimizer.minimize(self.error)

		self.sess.run(tf.global_variables_initializer())


	def train(self, images, emotions, batch_size, epochs, num_examples, batch_generator):

		for epoch in range(epochs):
			start_time = time.time()
			total_loss = 0
			batch_num = 0
			for img, lab in batch_generator(batch_size):

				img = self.preprocess(img)
				_, los, err = self.sess.run([self.train_op, self.loss, self.error], feed_dict = {self.x: img,
																								self.y: lab})
				total_loss += err
				batch_num += 1

			print("Epoch {0} Loss: {1:1f} Time: {2}".format(epoch+1, total_loss/batch_num, time.time()-start_time))

			# Save after every 10 epochs
			if epoch % self.save_epoch == 0:
				self.save_model()

	def save_model(self):
		saver = tf.train.Saver()
		saver.save(self.sess, self.save_file)

	def load_model(self):
		saver = tf.train.Saver()
		saver.restore(self.sess, tf.train.latest_checkpoint('./emotion_detection/simple_nn'))

	def preprocess(self, images):
		faces = []
		for img in images:
			faces.append(img.ravel())
		return faces

	def predict(self, image):
		pred = tf.nn.softmax(self.out)
		pred_emotion = tf.argmax(pred, 1)
		image = image.reshape(1, 2304)
		prediction, emot = self.sess.run([pred, pred_emotion], feed_dict={self.x: image}) 																	
		return prediction, emot

if __name__ == '__main__':
	main_training()