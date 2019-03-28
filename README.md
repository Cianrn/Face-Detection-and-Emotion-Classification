# Face-Detection-and-Emotion-Classification
Detect faces in an images and determine the persons emotion

The goal is to 1) identify faces in an images and 2) classify the emotion of the individual. Available with openCV, we use 
`cv2.CascadeClassifier` which is a cascade of boosted classifiers working with haar-like features to detect faces in an image. A 
Convolutional Neural Network is then trained to classify the emotion of the individual. The Fer2013 dataset is used consisting of over 35,000 grayscale images of faces and corresponding emotion as ground truth labels. 

The model is run in real-time using webcam images.

## Setup
**Required**:

- [x] Tensorflow
- [x] cv2
- [x] pandas
- [x] numpy

Model training is done within `emotion_model.py` and running the models on either webcam or for single images is done in `main_emotion_detection`
