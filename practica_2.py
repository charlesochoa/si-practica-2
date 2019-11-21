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

def distance(x1, x2):
	return (x1[0]-x2[0])**2 + (x1[1]-x2[1])**2

def isInsideEllipse(mx,x,axes):
	return x[0]> mx[0] - axes[0] and x[0]< mx[0] + axes[0] and x[1]> mx[1] - axes[1] and x[1]< mx[1] + axes[1]



np.random.seed(19680801)

# example data



# the histogram of the data
# n, bins, patches = ax.hist(x, num_bins, density=1)


# add a 'best fit' line
# y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
# ax.plot(bins, y, '--')
# ax.set_xlabel('Smarts')
# ax.set_ylabel('Probability density')
# ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')




# mypath = "Data2/"
# onlyfiles = sorted(glob.glob(mypath + "*.jpg"))
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

R = np.array([	[15, 0, 0, 0],
				[0, 30, 0, 0],
				[0, 0, 15, 0],
				[0, 0, 0, 30]])

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
velocities_x = np.array([])
velocities_y = np.array([])

scale_percent = 75 # percent of original size
axes = (0,0)
valid_detection = (0,0,0,0)
for i in onlyfiles :

	img = cv2.imread(i,cv2.IMREAD_COLOR)
	if first_time:
		width = int(img.shape[1] * scale_percent / 100)
		height = int(img.shape[0] * scale_percent / 100)
		dim = (width, height)

	img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


	winname = 'image'

	(rects, wghts) = hog.detectMultiScale(img, winStride=win, padding=pad, scale=sca)
	

	wTemp = 0
	d = 999999999
	if len(rects) > 0:
		print(dim)
		for i, (x,y,w,h) in enumerate(rects):
			secureColor = int(g*wghts[i]/1.4)
			cv2.rectangle(img, (x,y),(x+w,y+h),(0, secureColor,0))
			if first_time and wghts[i]>0.8:
				m_x =  np.array([x+w/2,y+h/2,0,0])
				valid_detection = (x+w/2,y+h/2,0,0)
				last_detection = np.array([x+w/2,y+h/2,0,0])
				first_time = False
			elif not first_time  and wghts[i]>1 and d > distance(last_detection,valid_detection) and isInsideEllipse(m_x,valid_detection,axes):
				d = distance(last_detection,(x,y))
				valid_detection = (x+w/2,y+h/2,0,0)
				wTemp = wghts[i]

		
	if not first_time:
		print(wTemp)
		m_x = prediction(A,m_x)
		P = covarianceUpdate(P, A, Q)
		print(m_x)
		axes = plot_ellipse.plot_ellipse(img, np.array([m_x[0],m_x[1]]).T,P[:2, :2],(0,0,r))
		if wTemp != 0:	
			Z = np.array([  valid_detection[0],
							valid_detection[1],
							(valid_detection[0]-last_detection[0])/countNoInput,
							(valid_detection[1]-last_detection[1])/countNoInput
						])
			velocities_x = np.append(velocities_x,[Z[2]])
			velocities_y = np.append(velocities_y,[Z[3]])
			plot_ellipse.plot_ellipse(img, np.array([Z[0],Z[1]]).T,R[:2, :2],(0,g,0))
			S = residualCovariance(H, P, R)
			K = kalmanGain(P, H, S)
			y = Z - H@m_x
			m_x = m_x + (K@y)
			P = (I - (K@H))@P
			axes = plot_ellipse.plot_ellipse(img, np.array([m_x[0],m_x[1]]).T,P[:2, :2],(b,0,0))	
			last_detection = valid_detection
			countNoInput= 0
		countNoInput+= 1
	
			
			
	
		
	cv2.namedWindow(winname)
	cv2.moveWindow(winname, 5,5) 
	
	cv2.imshow(winname,img)

	cv2.waitKey(1)
	
# Tweak spacing to prevent clipping of ylabel
# n, bins, patches = ax.hist(velocities_x, num_bins, density=1)
print(velocities_y)
fig, ax = plt.subplots()
n, bins, patches = ax.hist(velocities_y, len(velocities_y), density=3)
fig.tight_layout()
plt.show()


# print(veloc(predictions[0],predictions[1],0.5))
cv2.destroyAllWindows()

