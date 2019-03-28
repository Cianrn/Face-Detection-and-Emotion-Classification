from emotion_models import *
from utils_emotion import *

def main_webcam():

	## Dataset 
	data_path = '../fer2013.csv'
	images, emotions = load_fer2013(data_path)
	num_classes = len(emotions[0])
	emotion_labels = find_labels()
	network ='CNN'
	LEARNING_RATE = 1e-4

	## Load Network 
	if network == 'simple':
		network = BasicNN(num_classes=num_classes, learning_rate=LEARNING_RATE)
	elif network =='CNN':
		network = BasicCNN(num_classes=num_classes, learning_rate=LEARNING_RATE)

	network.load_model()

	## Detection Model
	detection_model_path = '../haarcascade_frontalface_default.xml'
	det_model = detection_model(detection_model_path)

	print("Starting webcam...")
	video_capture = cv2.VideoCapture(0)

	while True:

		ret, frame = video_capture.read()
		print(frame.shape)
		print(len(frame))

		gray = preprocess(frame, img='full', color='RGB')
		detected_faces = detect_faces(det_model, gray)
		num_faces = len(detected_faces)
		print("{0} faces detected".format(num_faces))

		for coordinates in detected_faces:
			x, y, w, h = coordinates
			x1, y1, x2, y2 = x, y, x+w, y+h

			rgb_face = frame[y1:y2, x1:x2]
			gray_face = preprocess(rgb_face, img='face', color='RGB')
			p, e = network.predict(gray_face)
			face_emotion = emotion_labels[e[0]]
			if face_emotion == 'happy':
				draw_bounding_box(coordinates, frame, color=(0, 255, 0))
				cv2.putText(frame, face_emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
			elif face_emotion == 'surprise':
				draw_bounding_box(coordinates, frame, color=(255, 0, 0))
				cv2.putText(frame, face_emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
			elif face_emotion == 'angry':
				draw_bounding_box(coordinates, frame, color=(0, 0, 255))
				cv2.putText(frame, face_emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
			elif face_emotion == 'neutral':
				draw_bounding_box(coordinates, frame, color=(100, 100, 100))
				cv2.putText(frame, face_emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
			else:
				draw_bounding_box(coordinates, frame, color=(150, 200, 200))
				cv2.putText(frame, face_emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 200, 200), 1)

			print(face_emotion)

		# Show webcam
		cv2.imshow('Video', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything is done, release the capture
	video_capture.release()
	cv2.destroyAllWindows()	


def main_single_images():
	test_or_train_images = 'test'

	if test_or_train_images:
		## predict on new images 
		test_image = load_single_image('../emotions.jpg') #Picture6.jpg') 
		test_image_gray = preprocess(test_image, img='full', color='RGB')

		detection_model_path = '../haarcascade_frontalface_default.xml'
		det_model = detection_model(detection_model_path)
		detected_faces = detect_faces(det_model, test_image_gray)
		num_faces = len(detected_faces)

		for coordinates in detected_faces:
			x, y, w, h = coordinates
			x1, y1, x2, y2 = x, y, x+w, y+h
			rgb_face = test_image[y1:y2, x1:x2]
			gray_face = preprocess(rgb_face, img='face', color='RGB')
			p, e = network.predict(gray_face)
			face_emotion = emotion_labels[e[0]]
			if face_emotion == 'happy':
				draw_bounding_box(coordinates, test_image, color=(0, 255, 0))
				cv2.putText(test_image, face_emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
			elif face_emotion == 'surprise':
				draw_bounding_box(coordinates, test_image, color=(0, 0, 255))
				cv2.putText(test_image, face_emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
			elif face_emotion == 'angry':
				draw_bounding_box(coordinates, test_image, color=(255, 0, 0))
				cv2.putText(test_image, face_emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
			elif face_emotion == 'neutral':
				draw_bounding_box(coordinates, test_image, color=(100, 100, 100))
				cv2.putText(test_image, face_emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
			else:
				draw_bounding_box(coordinates, test_image, color=(150, 200, 200))
				cv2.putText(test_image, face_emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 200), 1)

			print(face_emotion)

		plt.imshow(test_image)
		plt.show()
	else:
		## Predict on training images 
		for i in range(10):
			img = preprocess(images[i], color='GRAY')
			pred, emot = network.predict(img)
			predicted_emotion = emotion_labels[emot[0]]
			actual_emotion = emotion_labels[np.argmax(emotions[i])]
			print("predicted: ", predicted_emotion, "Actual: ", actual_emotion)
			## Visualising 
			cv2.putText(images[i], str(predicted_emotion), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 255, 0))
			plt.imshow(images[i], cmap='gray')
			plt.show()

if __name__ == '__main__':
	main_webcam()