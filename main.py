import cv2
import numpy as np
import face_detection.face_detection as face_detection
import os
import sys

def Image(path):
	face_detector = face_detection.FaceDetector()
	frame = cv2.imread(path)
	annotated_frame = face_detector.draw(frame)
	cv2.imshow('faces',annotated_frame)
	cv2.waitKey(0)

def Video():
	face_detector = face_detection.FaceDetector()
	cap = cv2.VideoCapture(0)
	while not cap.isOpened():
		cap = cv2.VideoCapture(0)
		cv2.waitKey(1000)
	label = ''
	cntr = 1
	while True:
		flag, frame = cap.read()
		if flag:
			frame = cv2.flip(frame,1)
			annotated_frame = face_detector.draw(frame)
			cv2.imshow('camera_feed',annotated_frame)
			#cv2.imshow('input',frame[100:300,400:600])
			
		else:
			cv2.waitKey(1000)
		k = cv2.waitKey(5)
		if k == 27:
			break

if __name__ == "__main__":
	Video()
	cv2.destroyAllWindows()
