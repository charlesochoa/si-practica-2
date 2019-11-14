import cv2
from os import listdir
from os.path import isfile, join
import time
import glob
import random
import numpy as np

def veloc(p1, p2, deltaT):
	return np.subtract(p1,p2)/deltaT



mypath = "Data/TestData/2012-04-02_120351/RectGrabber/"
onlyfiles = sorted(glob.glob(mypath + "*0.pgm"))
pt1 = ( 100 , 100)
pt2 = ( 200 , 200)

r =  255
g =  255
b = 0
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cal = (50, (4, 4), (16, 16), 1.2) # 0.8 overlap thresh
minw, win, pad, sca = cal

predictions = []


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


	
	
	for i, (x,y,w,h) in enumerate(rects):

		cv2.rectangle(img, (x,y),(x+w,y+h),(b,g,r))

		if wghts[i]> 0.5:
			# print(x,type(x))
			# print(y,type(y))
			# print(w,type(w))
			# print(h,type(h))
			predictions.append(np.array([x+ w/2,y+h/2]))
	
	print(rects, wghts)
	if len(predictions)>0 and len(predictions)<3:
		# print(wghts)
		# print(predictions[len(predictions)-1])
		continue
	elif len(predictions)==3:

		# print(wghts)
		# print(predictions[len(predictions)-1])
		# break
		continue
	# pick = non_max_suppression(rects, probs=None, overlapThresh=ovt)

	cv2.namedWindow(winname)
	cv2.moveWindow(winname, 5,5) 
	
	cv2.imshow(winname,img)

	cv2.waitKey(1)
	

# print(veloc(predictions[0],predictions[1],0.5))
cv2.destroyAllWindows()

