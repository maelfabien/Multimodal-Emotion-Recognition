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
    video_capture = cv2.VideoCapture(0)
    
    # Image shape
    shape_x = 48
    shape_y = 48
    input_shape = (shape_x, shape_y, 1)
    
    # We have 7 emotions
    nClasses = 7
    
    # Timer until the end of the recording
    end = 0
    
    # Count number of eye blinks (not used in model prediction)
    def eye_aspect_ratio(eye):
        
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    # Detect facial landmarks and return coordinates (not used in model prediction but in visualization)
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
    
    #  Zoom on the face of the person
    def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
        
        # Each face identified
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
    
    # Initiate Landmarks
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
    
    (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    
    # Load the pre-trained X-Ception model
    model = load_model('Models/video.h5')
    
    # Load the face detector
    face_detect = dlib.get_frontal_face_detector()
    
    # Load the facial landmarks predictor
    predictor_landmarks  = dlib.shape_predictor("Models/face_landmarks.dat")

    # Prediction vector
    predictions = []
    
    # Timer
    global k
    k = 0
    max_time = 15
    start = time.time()
    
    angry_0 = []
    disgust_1 = []
    fear_2 = []
    happy_3 = []
    sad_4 = []
    surprise_5 = []
    neutral_6 = []

    # Record for 45 seconds
    while end - start < max_time :
        
        k = k+1
        end = time.time()
        
        # Capture frame-by-frame the video_capture initiated above
        ret, frame = video_capture.read()
        
        # Face index, face by face
        face_index = 0
        
        # Image to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # All faces detected
        rects = face_detect(gray, 1)
        
        #gray, detected_faces, coord = detect_face(frame)
        
        
        # For each detected face
        for (i, rect) in enumerate(rects):
            
            # Identify face coordinates
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = gray[y:y+h,x:x+w]
            
            # Identify landmarks and cast to numpy
            shape = predictor_landmarks(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Zoom on extracted face
            face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))
            
            # Cast type float
            face = face.astype(np.float32)
            
            # Scale the face
            face /= float(face.max())
            face = np.reshape(face.flatten(), (1, 48, 48, 1))
            
            # Make Emotion prediction on the face, outputs probabilities
            prediction = model.predict(face)
            
            # For plotting purposes with Altair
            angry_0.append(prediction[0][0].astype(float))
            disgust_1.append(prediction[0][1].astype(float))
            fear_2.append(prediction[0][2].astype(float))
            happy_3.append(prediction[0][3].astype(float))
            sad_4.append(prediction[0][4].astype(float))
            surprise_5.append(prediction[0][5].astype(float))
            neutral_6.append(prediction[0][6].astype(float))
            
            # Most likely emotion
            prediction_result = np.argmax(prediction)
            
            # Append the emotion to the final list
            predictions.append(str(prediction_result))
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Top left : Put the ID of the face
            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw all the landmarks dots
            for (j, k) in shape:
                cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)
        
            # Add prediction probabilities on the top-left report
            cv2.putText(frame, "----------------",(40,100 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            cv2.putText(frame, "Emotional report : Face #" + str(i+1),(40,120 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            cv2.putText(frame, "Angry : " + str(round(prediction[0][0],3)),(40,140 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            cv2.putText(frame, "Disgust : " + str(round(prediction[0][1],3)),(40,160 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            cv2.putText(frame, "Fear : " + str(round(prediction[0][2],3)),(40,180 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            cv2.putText(frame, "Happy : " + str(round(prediction[0][3],3)),(40,200 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            cv2.putText(frame, "Sad : " + str(round(prediction[0][4],3)),(40,220 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            cv2.putText(frame, "Surprise : " + str(round(prediction[0][5],3)),(40,240 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            cv2.putText(frame, "Neutral : " + str(round(prediction[0][6],3)),(40,260 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            
            # Annotate main image with the emotion label
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
            
            # Eye Detection and Blink Count
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
            
            # Detect Nose and draw its contours
            nose = shape[nStart:nEnd]
            noseHull = cv2.convexHull(nose)
            cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)
            
            # Detect Mouth and draw its contours
            mouth = shape[mStart:mEnd]
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            
            # Detect Jaw and draw its contours
            jaw = shape[jStart:jEnd]
            jawHull = cv2.convexHull(jaw)
            cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)
            
            # Detect Eyebrows and draw its contours
            ebr = shape[ebrStart:ebrEnd]
            ebrHull = cv2.convexHull(ebr)
            cv2.drawContours(frame, [ebrHull], -1, (0, 255, 0), 1)

            ebl = shape[eblStart:eblEnd]
            eblHull = cv2.convexHull(ebl)
            cv2.drawContours(frame, [eblHull], -1, (0, 255, 0), 1)
    
        # Show number of faces captured
        cv2.putText(frame,'Number of Faces : ' + str(len(rects)),(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)
        
        # For flask, save image as t.jpg (rewritten at each step)
        cv2.imwrite('tmp/t.jpg', frame)
        
        # Yield the image at each step
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('tmp/t.jpg', 'rb').read() + b'\r\n')
        
        # Emotion mapping
        #emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
        
        # Once reaching the end, write the results to the personal file and to the overall file
        if end-start > max_time - 1 :
            with open("static/js/db/histo_perso.txt", "w") as d:
                d.write("density"+'\n')
                for val in predictions :
                    d.write(str(val)+'\n')
                
            with open("static/js/db/histo.txt", "a") as d:
                for val in predictions :
                    d.write(str(val)+'\n')
               
    
            rows = zip(angry_0,disgust_1,fear_2,happy_3,sad_4,surprise_5,neutral_6)

            import csv
            with open("static/js/db/prob.csv", "w") as d:
                writer = csv.writer(d)
                for row in rows:
                    writer.writerow(row)
         

            with open("static/js/db/prob_tot.csv", "a") as d:
                writer = csv.writer(d)
                for row in rows:
                    writer.writerow(row)
          
            K.clear_session()
            break

    video_capture.release()
# Clear session to allow user to do another test afterwards
#K.clear_session()


    # d.write(','.join(str(i) for i in angry_0)+'\n')
    # d.write(','.join(str(i) for i in disgust_1)+'\n')
    #d.write(','.join(str(i) for i in fear_2)+'\n')
    # d.write(','.join(str(i) for i in happy_3)+'\n')
    #  d.write(','.join(str(i) for i in sad_4)+'\n')
    #  d.write(','.join(str(i) for i in surprise_5)+'\n')
#  d.write(','.join(str(i) for i in neutral_6)+'\n')
