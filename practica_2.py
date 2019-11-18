import cv2
from os import listdir
from os.path import isfile, join
import time
import glob
import random
import numpy as np
import plot_ellipse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

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


np.random.seed(19680801)

# example data
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * np.random.randn(437)

num_bins = 50

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Smarts')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
# fig.tight_layout()
# plt.show()


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

Q = np.array([	[3, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 3, 0],
				[0, 0, 0, 1]])

R = np.array([	[1, 0, 0, 0],
				[0, 2, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 2]])

H = np.array([	[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1]])

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
scale_percent = 40 # percent of original size
# R = R/(scale_percent/100)
print(R)
for i in onlyfiles :

	img = cv2.imread(i,cv2.IMREAD_COLOR)
	if first_time:
		width = int(img.shape[1] * scale_percent / 100)
		height = int(img.shape[0] * scale_percent / 100)
		dim = (width, height)
		
	print(dim)
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
			secureColor = int(g*wghts[i]/1.4)
			cv2.rectangle(img, (x,y),(x+w,y+h),(0, secureColor,0))
			if wghts[i]>1:
				if first_time:
					m_x =  np.array([x+w/2,y+h/2,0,0])
					last_detection = np.array([x+w/2,y+h/2,0,0])
					Q = Q*width/20000
					first_time = False
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
		print(Q)
		
	
		
	cv2.namedWindow(winname)
	cv2.moveWindow(winname, 5,5) 
	
	cv2.imshow(winname,img)

	cv2.waitKey(1)
	
	

# print(veloc(predictions[0],predictions[1],0.5))
cv2.destroyAllWindows()

