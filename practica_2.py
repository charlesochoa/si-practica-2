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
	return A@x

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

Q = np.array([	[0.05, 0, 0, 0],
				[0, 0.01, 0, 0],
				[0, 0, 0.05, 0],
				[0, 0, 0, 0.01]])

R = np.array([	[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1]])

H = np.array([	[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1]])

# C = np.array([	[]])


# x + w/2 
# y + h/2

# y[0] + y[2]/2 = x[0]


# y[0] = x[0] - y[2]/2 
# y[1] = x[1] - y[3]/2 
# y[2] = (y-1[0] - y[0])/(deltaT*counterNoInput)
# y[3] = (y-1[1] - y[1])/(deltaT*counterNoInput)


I = np.array([	[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1]])


P = np.array([	[0, 0, 0, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 0]])





r =  255
g =  255
b = 255
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cal = (50, (4, 4), (16, 16), 1.2) # 0.8 overlap thresh
minw, win, pad, sca = cal

predictions = []

m_x = np.array([0,0,0,0]).T
m_x_old = np.array([0,0,0,0]).T

x_new = np.array([0,0]).T
first_time = True
countNoInput = 1
scale_percent = 50 # percent of original size
# R = R/(scale_percent/100)
print(R)
for i in onlyfiles :

	img = cv2.imread(i,cv2.IMREAD_COLOR)
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	# resize image
	img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

	# print(i)
	winname = 'image'
	# image = imutils.resize(img, width=100)
	(rects, wghts) = hog.detectMultiScale(img, winStride=win, padding=pad, scale=sca)
	
	# print(m_x)


	wTemp = 0

	if len(rects) > 0:
		for i, (x,y,w,h) in enumerate(rects):
			if first_time:
				m_x =  np.array([x+w/2,0,y+h/2,0])
				last_detection = np.array([x+w/2,0,y+h/2,0])
				first_time = False
			if wghts[i]>1:
				cv2.rectangle(img, (x,y),(x+w,y+h),(b,g,r))
				wTemp = wghts[i]
				break
	if not first_time:				
		if wTemp == 0:
			m_x = prediction(A,m_x)
			P = covarianceUpdate(P, A, Q)
			plot_ellipse.plot_ellipse(img, np.array([m_x[0],m_x[1]]).T,P[:2, :2],(0,0,r))	
			countNoInput+= 1
		else:
			Z = np.array([   x+w/2,
							 y+h/2,
							(x+w/2-last_detection[0])/countNoInput,
							(y+h/2-last_detection[1])/countNoInput
						])

			m_x = prediction(A,m_x)
			P = covarianceUpdate(P, A, Q)
			plot_ellipse.plot_ellipse(img, np.array([m_x[0],m_x[1]]).T,P[:2, :2],(0,0,r))	
			plot_ellipse.plot_ellipse(img, np.array([Z[0],Z[1]]).T,R[:2, :2],(0,g,0))	
			# print("Xt")
			# print(m_x)
			# print("Z")
			# print(Z)
			S = residualCovariance(H, P, R)
			K = kalmanGain(P, H, S)
			y = Z - H@m_x
			m_x = m_x + (K@y)
			P = (I - (K@H))@P
			# P = I
			plot_ellipse.plot_ellipse(img, np.array([m_x[0],m_x[1]]).T,P[:2, :2],(b,0,0))	
			last_detection = Z
			countNoInput= 1
		print(P)
		
	
		
	cv2.namedWindow(winname)
	cv2.moveWindow(winname, 5,5) 
	
	cv2.imshow(winname,img)

	cv2.waitKey(1)
	
	

# print(veloc(predictions[0],predictions[1],0.5))
cv2.destroyAllWindows()

