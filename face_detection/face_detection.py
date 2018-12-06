import numpy as np
import argparse
import cv2

class FaceDetector:
	def __init__(self,prototxt_path="./model/deploy.prototxt.txt",model_path="./model/res10_300x300_ssd_iter_140000.caffemodel",confidence=0.5):
		self.net = cv2.dnn.readNetFromCaffe(prototxt_path,model_path)
		self.confidence = confidence

	def detect(self,frame):
		detected_faces=[]
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),1.0,(300, 300), (104.0, 177.0, 123.0))
		self.net.setInput(blob)
		detections = self.net.forward()
		for i in range(0, detections.shape[2]):
			confidence = detections[0,0,i,2]
			if confidence > self.confidence:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				detected_faces.append({'start':(startX,startY),'end':(endX,endY),'confidence':confidence})
		return detected_faces
	
	def draw(self,frame):
		detected_faces = self.detect(frame)
		for face in detected_faces:
			cv2.rectangle(frame, face['start'], face['end'],(0, 255, 0), 2)
			y = face['start'][1] - 10 if face['start'][1] - 10 > 10 else face['start'][1] + 10
			cv2.putText(frame, "{:.4f}".format(face['confidence']), (face['start'][0],y) , cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
		return frame
