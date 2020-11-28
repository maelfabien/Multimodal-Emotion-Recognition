### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
from time import sleep
import re
import os
import requests
import argparse
from collections import OrderedDict
import math

### Image processing ###
import cv2
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage
import dlib
from imutils import face_utils

### Model ###
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def gen():
	"""
	Video streaming generator function.
	"""
	
	# Start video capute. 0 = Webcam, 1 = Video file, -1 = Webcam for Web
	video_capture = cv2.VideoCapture('/Users/nainatejani/Desktop/Faces/ClippedRight12.mp4')
	# # Image shape
	shape_x = 48
	shape_y = 48
	input_shape = (shape_x, shape_y, 1)
	
	# # We have 7 emotions
	nClasses = 7
	
	# # Timer until the end of the recording
	# end = 0
	
	# # Count number of eye blinks (not used in model prediction)
	def eye_aspect_ratio(eye):
		
		A = distance.euclidean(eye[1], eye[5])
		B = distance.euclidean(eye[2], eye[4])
		C = distance.euclidean(eye[0], eye[3])
		ear = (A + B) / (2.0 * C)
		
		return ear
	# # Detect facial landmarks and return coordinates (not used in model prediction but in visualization)
	def detect_face(frame):
		
		#Cascade classifier pre-trained model
		cascPath = 'Models/face_landmarks.dat'
		faceCascade = cv2.CascadeClassifier(cascPath)
		
		#BGR -> Gray conversion
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		#Cascade MultiScale classifier
		detected_faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6,
													  minSize=(shape_x, shape_y),
													  flags=cv2.CASCADE_SCALE_IMAGE)
		coord = []
													  
		for x, y, w, h in detected_faces :
			if w > 100 :
				# Square around the landmarks
				sub_img=frame[y:y+h,x:x+w]
				# Put a rectangle around the face
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255,255),1)
				coord.append([x,y,w,h])
														  
		return gray, detected_faces, coord
	# #  Zoom on the face of the person
	def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
		
	#     # Each face identified
		gray = faces[0]
		
		# ID of each face identifies
		detected_face = faces[1]
		
		new_face = []
		
		for det in detected_face :
			# Region in which the face is detected
			# x, y represent the starting point, w the width (moving right) and h the height (moving up)
			x, y, w, h = det
			
			#Offset coefficient (margins), np.floor takes the lowest integer (delete border of the image)
			horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
			vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
			
			# Coordinates of the extracted face
			extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
			
			#Zoom on the extracted face
			new_extracted_face = zoom(extracted_face, (shape_x / extracted_face.shape[0],shape_y / extracted_face.shape[1]))
			
			# Cast type to float
			new_extracted_face = new_extracted_face.astype(np.float32)
			
			# Scale the new image
			new_extracted_face /= float(new_extracted_face.max())
			
			# Append the face to the list
			new_face.append(new_extracted_face)
		
		return new_face

	# # Load the pre-trained X-Ception model
	model = load_model('/Users/nainatejani/Desktop/Faces/video.h5')
	# # Load the face detector
	face_detect = dlib.get_frontal_face_detector()
	# # Load the facial landmarks predictor
	predictor_landmarks  = dlib.shape_predictor("/Users/nainatejani/Desktop/Faces/face_landmarks.dat")
	# # Prediction vector
	predictions = []
	# # # Timer
	global k
	k = 0
	
	angry_0 = []
	disgust_1 = []
	fear_2 = []
	happy_3 = []
	sad_4 = []
	surprise_5 = []
	neutral_6 = []

	if (video_capture.isOpened()== False): 
		print("Error opening video stream or file")

	count = 0
	frameRate = video_capture.get(5) #frame rate
	# print("at least I arrived here")
	# # Record for 45 seconds
	print(frameRate, "frameRate")
	while video_capture.isOpened() :
		count+=1
		k = k+1
		frameId = video_capture.get(1)

	    # Capture frame-by-frame the video_capture initiated above
		ret, frame = video_capture.read()
		if ret != True:
  			print("OOF BREAKING")
  			break;

		# Image to gray scale
		if (frameId % 3 == 0):
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# 		# All faces detected
			rects = face_detect(gray, 1)
			if len(rects) == 0:
				predictions.append('')
				angry_0.append(0)
				disgust_1.append(0)
				fear_2.append(0)
				happy_3.append(0)
				sad_4.append(0)
				surprise_5.append(0)
				neutral_6.append(0)
					
				print("NO FACE FOUND")
		#     # For each detected face
			for (i, rect) in enumerate(rects[-1:]):
				(x, y, w, h) = face_utils.rect_to_bb(rect)
				# print(x,y,w,h, "dimensions")
				# print(y, y+h, "y", "y+h")
				# print(x, x+w, "x", "x+w")

				# print("entered for loop")
				# Identify landmarks and cast to numpy
				shape = predictor_landmarks(gray, rect)
				shape = face_utils.shape_to_np(shape)
				# Zoom on extracted face
				# print(face, "face")
				# print(face.shape, "shape of face")
				if x < 0:
  					x = 0
				if y < 0:
  					y = 0
				if (y+h) >= gray.shape[1]:
					h = gray.shape[1] - 1 + y
				if (x+w) >= gray.shape[0]:
					w = gray.shape[0]-1+x
				face = gray[y:y+h,x:x+w]
				face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))
				# Cast type float
				face = face.astype(np.float32)
				# Scale the face

				face /= float(face.max())
				face = np.reshape(face.flatten(), (1, 48, 48, 1))
				# Make Emotion prediction on the face, outputs probabilities
				prediction = model.predict(face)
		#         # For plotting purposes with Altair
				angry_0.append(prediction[0][0].astype(float))
				disgust_1.append(prediction[0][1].astype(float))
				fear_2.append(prediction[0][2].astype(float))
				happy_3.append(prediction[0][3].astype(float))
				sad_4.append(prediction[0][4].astype(float))
				surprise_5.append(prediction[0][5].astype(float))
				neutral_6.append(prediction[0][6].astype(float))
				
		#         # Most likely emotion
				prediction_result = np.argmax(prediction)
				
		#         # Append the emotion to the final list
				predictions.append(str(prediction_result))
				
		#     # Emotion mapping
			emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
			# print(predictions,"PREDICTIONS")
		#     # Once reaching the end, write the results to the personal file and to the overall file
			with open("/Users/nainatejani/Desktop/Faces/histo_perso.txt", "w") as d:
				d.write("density"+'\n')
				for val in predictions :
					d.write(str(val)+'\n')
				
			with open("/Users/nainatejani/Desktop/Faces/histo.txt", "a") as d:
				for val in predictions :
					d.write(str(val)+'\n')
				

			rows = zip(angry_0,disgust_1,fear_2,happy_3,sad_4,surprise_5,neutral_6)

			import csv
			print("about to write to csv", frameId/frameRate)
			with open("/Users/nainatejani/Desktop/Faces/prob.csv", "w") as d:
				writer = csv.writer(d)
				for row in rows:
					writer.writerow(row)
			

			with open("/Users/nainatejani/Desktop/Faces/prob_tot.csv", "a") as d:
				writer = csv.writer(d)
				for row in rows:
					writer.writerow(row)
			K.clear_session()


	print(count, "count")
	video_capture.release()
	print("count")
	print(count)

import altair as alt
def plot():
	df_altair = pd.read_csv('/Users/nainatejani/Desktop/results/facial_data.csv', header=None, index_col=None).reset_index()
	df_altair.columns = ['Time', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

	
	angry = alt.Chart(df_altair).mark_line(color='orange', strokeWidth=2).encode(
	   x='Time:Q',
	   y='Angry:Q',
	   tooltip=["Angry"]
	)

	disgust = alt.Chart(df_altair).mark_line(color='red', strokeWidth=2).encode(
		x='Time:Q',
		y='Disgust:Q',
		tooltip=["Disgust"])


	fear = alt.Chart(df_altair).mark_line(color='green', strokeWidth=2).encode(
		x='Time:Q',
		y='Fear:Q',
		tooltip=["Fear"])


	happy = alt.Chart(df_altair).mark_line(color='blue', strokeWidth=2).encode(
		x='Time:Q',
		y='Happy:Q',
		tooltip=["Happy"])


	sad = alt.Chart(df_altair).mark_line(color='black', strokeWidth=2).encode(
		x='Time:Q',
		y='Sad:Q',
		tooltip=["Sad"])


	surprise = alt.Chart(df_altair).mark_line(color='pink', strokeWidth=2).encode(
		x='Time:Q',
		y='Surprise:Q',
		tooltip=["Surprise"])


	neutral = alt.Chart(df_altair).mark_line(color='brown', strokeWidth=2).encode(
		x='Time:Q',
		y='Neutral:Q',
		tooltip=["Neutral"])


	chart = (angry + disgust + fear + happy + sad + surprise + neutral).properties(
	width=1000, height=400, title='Probability of each emotion over time')
	alt.renderers.enable('mimetype')
	chart.save('/Users/nainatejani/Desktop/chart1.html')
	
if __name__ == '__main__':	
	# plot()
	try :
	    # Response is used to display a flow of information
	  print("here")

	  gen()
	  print("here again")
	#return Response(stream_template('video.html', gen()))
	except :
	    pass
