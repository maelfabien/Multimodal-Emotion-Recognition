### General imports ###
import numpy as np
import pandas as pd
import cv2

from time import time
from time import sleep
import re
import os

import argparse
from collections import OrderedDict
import matplotlib.animation as animation

### Image processing ###
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage

import dlib
from __future__ import division

from tensorflow.keras.models import load_model
from imutils import face_utils

import requests

class ModelPredict() :
	
	def __init__(self, path, model, image) :
		self.image = image
		self.path = path
		self.model = model

	def detect_face(frame):
	    
	    global shape_x
	    global shape_y
	    global input_shape
	    global nClasses

	    shape_x = 48
		shape_y = 48
		input_shape = (shape_x, shape_y, 1)
		nClasses = 7

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
	            sub_img=frame[y:y+h,x:x+w]
	            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255,255),1)
	            coord.append([x,y,w,h])
	    
	    return gray, detected_faces, coord

	def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
	    gray = faces[0]
	    detected_face = faces[1]
	    
	    new_face = []
	    
	    for det in detected_face :
	        #Region dans laquelle la face est détectée
	        x, y, w, h = det
	        #X et y correspondent à la conversion en gris par gray, et w, h correspondent à la hauteur/largeur
	    
	        #Offset coefficient, np.floor takes the lowest integer (delete border of the image)
	        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
	        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))

	        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	        #gray transforme l'image
	        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
	    
	        #Zoom sur la face extraite
	        new_extracted_face = zoom(extracted_face, (shape_x / extracted_face.shape[0],shape_y / extracted_face.shape[1]))
	        #cast type float
	        new_extracted_face = new_extracted_face.astype(np.float32)
	        #scale
	        new_extracted_face /= float(new_extracted_face.max())
	        #print(new_extracted_face)
	    
	        new_face.append(new_extracted_face)
	    
	    return new_face

	def start_live() : 

		(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
		(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
		(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

		(eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
		(ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

		model = load_model('Models/video.h5')
		#Lancer la capture video
		video_capture = cv2.VideoCapture(0)

		while True:
		    # Capture frame-by-frame
		    ret, frame = video_capture.read()
		    
		    face_index = 0
		    
		    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		    rects = face_detect(gray, 1)
		    #gray, detected_faces, coord = detect_face(frame)

		    for (i, rect) in enumerate(rects):

		        shape = predictor_landmarks(gray, rect)
		        shape = face_utils.shape_to_np(shape)
		        
		        # Identify face coordinates
		        (x, y, w, h) = face_utils.rect_to_bb(rect)
		        face = gray[y:y+h,x:x+w]
		        
		        #Zoom on extracted face
		        face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))
		        
		        #Cast type float
		        face = face.astype(np.float32)
		        
		        #Scale
		        face /= float(face.max())
		        face = np.reshape(face.flatten(), (1, 48, 48, 1))
		        
		        #Make Prediction
		        prediction = model.predict(face)
		        prediction_result = np.argmax(prediction)
		        
		        # Rectangle around the face
		        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		    
		        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		 
		        for (j, k) in shape:
		            cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)
		            
		        # 1. Add prediction probabilities
		        cv2.putText(frame, "----------------",(40,100 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
		        cv2.putText(frame, "Emotional report : Face #" + str(i+1),(40,120 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
		        cv2.putText(frame, "Angry : " + str(round(prediction[0][0],3)),(40,140 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
		        cv2.putText(frame, "Disgust : " + str(round(prediction[0][1],3)),(40,160 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
		        cv2.putText(frame, "Fear : " + str(round(prediction[0][2],3)),(40,180 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
		        cv2.putText(frame, "Happy : " + str(round(prediction[0][3],3)),(40,200 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
		        cv2.putText(frame, "Sad : " + str(round(prediction[0][4],3)),(40,220 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
		        cv2.putText(frame, "Surprise : " + str(round(prediction[0][5],3)),(40,240 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
		        cv2.putText(frame, "Neutral : " + str(round(prediction[0][6],3)),(40,260 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
		        
		        # 2. Annotate main image with a label
		        if prediction_result == 0 :
		            cv2.putText(frame, "Angry",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		        elif prediction_result == 1 :
		            cv2.putText(frame, "Disgust",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		        elif prediction_result == 2 :
		            cv2.putText(frame, "Fear",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		        elif prediction_result == 3 :
		            cv2.putText(frame, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		        elif prediction_result == 4 :
		            cv2.putText(frame, "Sad",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		        elif prediction_result == 5 :
		            cv2.putText(frame, "Surprise",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		        else :
		            cv2.putText(frame, "Neutral",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		                   
		        # 3. Eye Detection and Blink Count
		        leftEye = shape[lStart:lEnd]
		        rightEye = shape[rStart:rEnd]
		        
		        # Compute Eye Aspect Ratio
		        leftEAR = eye_aspect_ratio(leftEye)
		        rightEAR = eye_aspect_ratio(rightEye)
		        ear = (leftEAR + rightEAR) / 2.0
		            
		        # And plot its contours
		        leftEyeHull = cv2.convexHull(leftEye)
		        rightEyeHull = cv2.convexHull(rightEye)
		        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		        
		        # 4. Detect Nose
		        nose = shape[nStart:nEnd]
		        noseHull = cv2.convexHull(nose)
		        cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

		        # 5. Detect Mouth
		        mouth = shape[mStart:mEnd]
		        mouthHull = cv2.convexHull(mouth)
		        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		            
		        # 6. Detect Jaw
		        jaw = shape[jStart:jEnd]
		        jawHull = cv2.convexHull(jaw)
		        cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)
		            
		        # 7. Detect Eyebrows
		        ebr = shape[ebrStart:ebrEnd]
		        ebrHull = cv2.convexHull(ebr)
		        cv2.drawContours(frame, [ebrHull], -1, (0, 255, 0), 1)
		        ebl = shape[eblStart:eblEnd]
		        eblHull = cv2.convexHull(ebl)
		        cv2.drawContours(frame, [eblHull], -1, (0, 255, 0), 1)
		        
		    cv2.putText(frame,'Number of Faces : ' + str(len(rects)),(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)
		    cv2.imshow('Video', frame)
		    
		    if cv2.waitKey(1) & 0xFF == ord('q'):
		        break

		# When everything is done, release the capture
		video_capture.release()
		cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) is not 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        # TAKE ID FROM SHAREABLE LINK
        file_id = sys.argv[1]
        # DESTINATION FILE ON YOUR DISK
        destination = sys.argv[2]
        download_file_from_google_drive(file_id, destination)

