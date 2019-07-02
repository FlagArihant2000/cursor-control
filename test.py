import numpy as np
import cv2
import dlib
from imutils import face_utils

def MAR(point1,point2,point3,point4,point5,point6,point7,point8):
	mar = (dst(point1,point2) + dst(point3,point4) + dst(point5,point6))/(3.0*dst(point7,point8))
	return mar

#The function is used for calculation of EAR for an eye
def EAR(point1,point2,point3,point4,point5,point6):
	ear = (dst(point2,point6) + dst(point3,point5))/(2*dst(point1,point4))*1.0
	return ear

def dst(point1, point2):
	distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
	return distance

img = cv2.imread('a.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


p = "shape_predictor.dat"
detector = dlib.get_frontal_face_detector() # Returns a default face detector object
predictor = dlib.shape_predictor(p) # Outputs a set of location points that define a pose of the object. (Here, pose of the human face)
rects = detector(gray,0)

for (i,rect) in enumerate(rects):
	shape = predictor(gray,rect)
	shape = face_utils.shape_to_np(shape)
	lefteye = EAR(shape[36],shape[37],shape[38],shape[39],shape[40],shape[41])
	righteye = EAR(shape[42],shape[43],shape[44],shape[45],shape[46],shape[47])
	mar = MAR(shape[50],shape[58],shape[51],shape[57],shape[52],shape[56],shape[48],shape[54])
print(lefteye,righteye,mar) 
print(len(shape))
for (x,y) in shape:
	cv2.circle(img,(x,y),2,(0,255,0),-1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
	


