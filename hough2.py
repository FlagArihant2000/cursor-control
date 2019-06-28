import numpy as np
import cv2
import math as m
from matplotlib import pyplot as plt



"""
Steps for Hough Transform:
1) Edge Detection
2) Mapping of edge points to the Hough Space and storage in accumulator
3) Interpretation of the accumulator to yield lines of infinite length.(Done by thresholding)
4) Conversion of infinite to finite lines.
"""
def dist(point1,point2):
	distance = m.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
	return distance 



img = cv2.imread('X.jpg',0)

edges = cv2.Canny(img,100,200) # Step 1 done
width,height = img.shape
black = np.zeros((width,height,3),np.uint8)
print(width,height)
theta = -90
i = 0
j = 0
counter = 0
thetalist = []
rholist = []
while i < height:
	while j < width:
		if(edges[j,i] != 0):
			while theta >= -90 and theta <= 90:
				thetaa = np.deg2rad(theta)
				cos_t = np.cos(thetaa)
				sin_t = np.sin(thetaa)
				rho = i*cos_t + j*sin_t
				print(theta,rho)
				#plt.plot(theta,rho)
				if(counter <= 1000):
					counter = counter + 1
				else:
					break
				theta = theta + 1
		else:
			pass
		if counter >= 1000:
			break
		theta = -90
		j = j + 1
	if counter >= 1000:
		break
	i = i + 1
	j = 0

#print(thetalist)
#print(rholist)
cv2.imshow('canny',edges)
cv2.imshow('black',black)
#cv2.imshow('image',img)
plt.plot(thetalist,rholist)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
