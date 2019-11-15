import cv2
from os import listdir
from os.path import isfile, join
import time
import glob
import random
import numpy as np
import plot_ellipse

def veloc(p1, p2, deltaT):
	return np.subtract(p1,p2)/deltaT

def prediction(A, x):
	return A.dot(x)

def covarianceUpdate(P, A, Q):
	return A@P@A.T + Q

def residualCovariance(H,P,R):
	return H*P*H.T + R

def kalmanGain(P,H,S):
	return (P*H.T) * np.linalg.pinv(S)


mypath = "Data/TestData/2012-04-02_120351/RectGrabber/"
onlyfiles = sorted(glob.glob(mypath + "*0.pgm"))
pt1 = ( 100 , 100)
pt2 = ( 200 , 200)
deltaT = 1
A = np.array([	[1, 0, deltaT, 0],
				[0, 1, 0, deltaT],
				[0, 0, 1, 0],
				[0, 0, 0, 1]])

C = np.array([	[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1]])

Q = np.array([	[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1]])





r =  255
g =  255
b = 0
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cal = (50, (4, 4), (16, 16), 1.2) # 0.8 overlap thresh
minw, win, pad, sca = cal

predictions = []

m_x = np.array([-1,-1,-1,-1]).T

for i in onlyfiles :

	img = cv2.imread(i,cv2.IMREAD_COLOR)
	scale_percent = 70 # percent of original size
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	# resize image
	img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

	# print(i)
	winname = 'image'
	# image = imutils.resize(img, width=100)
	(rects, wghts) = hog.detectMultiScale(img, winStride=win, padding=pad, scale=sca)
	




	
	if len(rects) > 0:
		for i, (x,y,w,h) in enumerate(rects):
			cv2.rectangle(img, (x,y),(x+w,y+h),(b,g,r))
			if wghts[i]>1:
				m_x = np.array([x+w/2,0,y+h/2,0]).T

	elif len(rects) == 0:
		m_x = prediction(A,m_x)
		
	cv2.namedWindow(winname)
	cv2.moveWindow(winname, 5,5) 
	
	cv2.imshow(winname,img)

	cv2.waitKey(1)
	

# print(veloc(predictions[0],predictions[1],0.5))
cv2.destroyAllWindows()

