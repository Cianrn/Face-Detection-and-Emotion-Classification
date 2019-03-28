import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.utils import shuffle

################### Utility Functions ######################

def detection_model(model_path):
	"""
	A cascade of boosted classifiers working with haar-like features. 
	:model_path: path to model trained by haar features
	"""
	return cv2.CascadeClassifier(model_path)

def detect_faces(det_model, image):
	"""
	Detects objects of different sizes in the input image. The detected objects are 
	returned as a list of rectangles.
	"""
	return det_model.detectMultiScale(image, scaleFactor=1.05, minNeighbors=5)

def draw_bounding_box(coordinates, image, color):
	"""
	Draw boxes around detected faces
	"""
	x, y, w, h = coordinates
	cv2.rectangle(image, (x, y), (x+w, y+h), color, 2) # Bottom left and top right coordinates:

def load_single_image(path):
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

def load_fer2013(data_path):
	data = pd.read_csv(data_path) # Data columns are emotion, pixels, Usage with 
	pixels = data['pixels'].tolist()
	width, height = 48, 48 # row of 2304 number in excel
	image_size = (48, 48)
	faces = []
	for row in pixels:
		face = [int(pixel) for pixel in row.split(' ')]
		face = np.asarray(face).reshape(width, height) # reshape into 48x48 image
		face = cv2.resize(face.astype('uint8'), image_size)
		faces.append(face.astype('float32'))
	faces = np.asarray(faces)
	emotions = pd.get_dummies(data['emotion']).as_matrix() # converts emotion into dummy variables e.g. [1 0 0 0 0 0 0]
	return faces, emotions

def find_labels(dataset='fer2013'):
	if dataset == 'fer2013':
		labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
					4: 'sad', 5: 'surprise', 6: 'neutral'}
	return labels

def gen_batch_function(images, emotions):
	"""
	Constructs a generator to pass batch_size images and labels for training
	:images: list of all images in dataset
	:emotion: list of all corresponding ground truth labels
	"""
	print("Generator made...")

	def batch_generator(batch_size=32):

		x, y = shuffle(images, emotions)
		for batch in range(0, len(x), batch_size):

			batch_x, batch_y = [], []

			for i, l in zip(x[batch:batch+batch_size], y[batch:batch+batch_size]):

				batch_x.append(i)
				batch_y.append(l)

			batch_x = preprocess(np.asarray(batch_x), img='full', color='gray', multiple=True)
			yield batch_x, np.array(batch_y)

	return batch_generator

def preprocess(image, img='full', color='GRAY', multiple=False):
	"""
	:image: input of single image or multiple images
	:img: full image or single detected face within image
	:color: graysace or rgb image
	"""
	if multiple:
		image = np.reshape(image, [len(image), 48, 48, 1])
	else:
		if color == 'RGB':
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			if img == 'face':
				# to match network input specifications
				image = cv2.resize(image, (48, 48))
				image = np.reshape(image, [1, 48, 48, 1])
		else:
			image = image.reshape([1, 48, 48, 1])

	return image






