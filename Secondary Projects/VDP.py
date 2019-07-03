import numpy as np
import cv2

cap = cv2.VideoCapture(0)
contour_sizes = 0
pad = np.ones((480,640,3),np.uint8)*255
kernel = np.ones((5,5),np.uint8)

def nothing(x):
	pass
cv2.namedWindow('color')
colour = np.zeros((100,100,3),np.uint8)
cv2.createTrackbar('R','color',0,255,nothing)
cv2.createTrackbar('G','color',0,255,nothing)
cv2.createTrackbar('B','color',0,255,nothing)
cv2.createTrackbar('S','color',3,10,nothing)
while True:
	ret,frame = cap.read()
	try:
		frame = cv2.flip(frame,1)
		r = cv2.getTrackbarPos('R','color')
		g = cv2.getTrackbarPos('G','color')
		b = cv2.getTrackbarPos('B','color')
		s = cv2.getTrackbarPos('S','color')
		colour[:] = [b,g,r]
		hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		blur1 = cv2.GaussianBlur(hsv,(5,5),100)
		lower_color = np.array([50,100,100])
		upper_color = np.array([70,255,255])
		mask = cv2.inRange(blur1,lower_color,upper_color)
		mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
		mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
		contours,hierarchy = cv2.findContours(mask,1,2) 
		contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
		biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
		(x,y),radius = cv2.minEnclosingCircle(biggest_contour)
		center = (int(x), int(y))
		radius = int(radius)
		img = cv2.circle(frame, center, radius, (255,0,0), 2)
		M = cv2.moments(mask)
		cX = int(M["m10"]/M["m00"])
		cY = int(M["m01"]/M["m00"])
		cv2.circle(frame,(cX,cY),5,(0,0,255),-1)
		cv2.circle(pad,(cX,cY),s,(b,g,r),-1)
		cv2.imshow('frame',frame)
		cv2.imshow('pad',pad)
		#cv2.imshow('hsv',hsv)
		#cv2.imshow('mask',mask)
		print('Press Esc to exit')
		if cv2.waitKey(1) & 0xFF == 27:
			print('Goodbye') 
			break
	except ValueError:
		print('Paused')
cap.release() 
cv2.destroyAllWindows() 
