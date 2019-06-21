#importing imutils and dlib for determination of face landmarks
from imutils import face_utils 
import dlib
#importing cv2(OpenCV) for computer vision
import cv2
#importing pyautogui for integrating the code with mouse
import pyautogui as pag
#importing numpy to perform array operations
import numpy as np
#The library is basically used for plotting the difference in EAR for different cases, while caliberating
from matplotlib import pyplot as plt
#importing time for caliberating the code for a particular time
import time
#importing math for basic mathematical operations
import math as m
#scipy used here for smoothening the matplotlib curves
from scipy.interpolate import spline

#The function is used for calculation of EAR for an eye
def EAR(point1,point2,point3,point4,point5,point6):
	ear = (dst(point2,point6) + dst(point3,point5))/(2*dst(point1,point4))*1.0
	return ear
# The function is used for calculating the distance between the two points. This is primarily used for calculating EAR
def dst(point1, point2):
	distance = m.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
	return distance

#The function is used for calculating the angle between the line made by the nose tip and centre of reference circle and the horizontal line passing through the centre of the reference circle
def angle(point1):
	# In OpenCV, since the x and y coordinates start from the top right corner of the image or the image is in the fourth quadrant, the conventions applied are in reverse.
	slope12 = (point1[1] - 250)/(point1[0] - 250)*1.0
	agle = -1.0*m.atan(slope12)
	return agle

#Initialization of numpy arrays for storing EAR differences as well as the time (in seconds) in which it was recorded
lclick = np.array([])
rclick = np.array([])
eyeopen = np.array([])
t1 = np.array([])
t2 = np.array([])
t3 = np.array([])

#importing the .dat file into the variable p, which will be used for the calculation of facial landmarks
p = "shape_predictor.dat"
detector = dlib.get_frontal_face_detector() # Returns a default face detector object
predictor = dlib.shape_predictor(p) # Outputs a set of location points that define a pose of the object. (Here, pose of the human face)

#The first snippet of code is basically for calibration of the EARdifference for left as well as the right eye
cap = cv2.VideoCapture(0)
#font used for putting text on the screen(will be used in sometime)
font = cv2.FONT_HERSHEY_SIMPLEX
#Standard reference of time is UNIX time
currenttime = time.time()#captures the current UNIX time
while(time.time() - currenttime <= 23): #The calibration code will run for 23 seconds.
	ret,image = cap.read()
	image = cv2.flip(image,1)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray)
	rects = detector(gray,0)
	for (i,rect) in enumerate(rects):
		shape = predictor(gray,rect)
		shape = face_utils.shape_to_np(shape)
		lefteye = EAR(shape[36],shape[37],shape[38],shape[39],shape[40],shape[41])
		righteye = EAR(shape[42],shape[43],shape[44],shape[45],shape[46],shape[47])
		EARdiff = (lefteye - righteye)*100
		elapsedTime = time.time() - currenttime
		if elapsedTime < 5.0: # recording the phase where the user is supposed to be ready for the calibration step by sitting in a comfortable position
			cv2.putText(image,'Calibration to Begin..',(0,100), font, 1,(0,0,0),2,cv2.LINE_AA)
		elif elapsedTime > 5.0 and elapsedTime < 10.0: # recording the phase for both eyes open to set reference point
			cv2.putText(image,'Keep both eyes open',(0,100), font, 1,(0,0,0),2,cv2.LINE_AA)
			eyeopen = np.append(eyeopen,[EARdiff])
			t1 = np.append(t1,[elapsedTime])
		elif elapsedTime > 10.0 and elapsedTime < 15.0: # recording the phase for only left eye closed
			cv2.putText(image,'Close left eye',(0,100), font, 1,(0,0,0),2,cv2.LINE_AA)
			lclick = np.append(lclick,[EARdiff])
			t2 = np.append(t2,[elapsedTime])
		elif elapsedTime > 15.0 and elapsedTime < 20.0: # recording the phase for only right eye closed
			cv2.putText(image,'Open left eye and close right eye',(0,100),font,1,(0,0,0),2,cv2.LINE_AA)
			rclick = np.append(rclick,[EARdiff])
			t3 = np.append(t3,[elapsedTime])
		else: # no recording done in this phase. It is just a 3 second lag in the code
			pass 
		for (x,y) in shape: # prints facial landmarks on the face
			cv2.circle(image,(x,y),2,(0,255,0),-1)
		cv2.imshow('image',image) # display of image
	if cv2.waitKey(5) & 0xff == 27: 
		break

# Plotting the recorded points when both eyes are open
plt.subplot(2,2,1)
t1_smooth = np.linspace(min(t1),max(t1),25)
eyeopen_smooth = spline(t1,eyeopen,t1_smooth)
plt.title('Both Eyes Open')
plt.plot(t1_smooth,eyeopen_smooth)

# Plotting the recorded points when left eye is closed
plt.subplot(2,2,2)
t2_smooth = np.linspace(min(t2),max(t2),25)
lclick_smooth = spline(t2,lclick,t2_smooth)
plt.title('Left click')
plt.plot(t2_smooth,lclick_smooth)

# Plotting the recorded points when right eye is closed
plt.subplot(2,2,3)
t3_smooth = np.linspace(min(t3),max(t3),25)
rclick_smooth = spline(t3,rclick,t3_smooth)
plt.title('Right click')
plt.plot(t3_smooth,rclick_smooth)

plt.show() # Display of graph. Press any key to exit the graph
cap.release()
cv2.destroyAllWindows()


# The second snippet of code consists of the main code, where all the cursor controlling using face gestures will take place
cap = cv2.VideoCapture(0)
openeyes = np.mean(eyeopen) # Calculates mean of the recorded values for the case when both eyes are opened.
leftclick = np.mean(lclick) - 1.5# Calculates mean of the recorded values for left click. Subtracting a constant to make it a bit more flexible
rightclick = np.mean(rclick) + 1.5 # Calculates mean of the recorded values for right click. Adding a constant to make it a bit more flexible
print("Left click value = "+str(leftclick)) # Prints the result
print("Right click value = "+str(rightclick)) # Prints the result
while(True):
	try: 
	    # Getting out image by webcam
		_, image = cap.read() 
		image=cv2.flip(image,1)
	    # Converting the image to gray scale image
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)
		cv2.circle(image,(250,250),50,(0,0,255),2)
	    # Get faces into webcam's image
		rects = detector(gray, 0)
	    # For each detected face, find the landmark.
		for (i, rect) in enumerate(rects):
		# Make the prediction and transfom it to numpy array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			[h,k] = shape[33]
			lefteye = EAR(shape[36],shape[37],shape[38],shape[39],shape[40],shape[41]) # Calculation of the EAR for left eye
			righteye = EAR(shape[42],shape[43],shape[44],shape[45],shape[46],shape[47]) # Calculation of the EAR for right eye
			EARdiff = (lefteye - righteye)*100 # Calculating the difference in EAR in percentage
			if EARdiff < leftclick: # Left click will be initiated if the EARdiff is less than the leftclick calculated during calibration
				pag.click(button = 'left') 
			elif EARdiff > rightclick: # Right click will be initiated if the EARdiff is more than the rightclick calculated during calibration
				pag.click(button = 'right')
				
					
			cv2.line(image,(250,250),(h,k),(0,0,0),1) # Draws a line between the tip of the nose and the centre of reference circle
			# Controls the move of mouse according to the position of the found nose.
		if((h-250)**2 + (k-250)**2 - 50**2 > 0):
			a = angle(shape[33]) # Calculates the angle
			if h > 250: # The below conditions set the conditions for the mouse to move and that too in any direction we desire it to move to.
				pag.moveTo(pag.position()[0]+(10*m.cos(-1.0*a)),pag.position()[1]+(10*m.sin(-1.0*a)),duration = 0.01)
			else:
				pag.moveTo(pag.position()[0]-(10*m.cos(-1.0*a)),pag.position()[1]-(10*m.sin(-1.0*a)),duration = 0.01)
		
	    	
		# Draw on our image, all the finded cordinate points (x,y) 
		for (x, y) in shape:
			cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
		cv2.circle(image,(h,k),2,(255,0,0),-1)
		cv2.circle(image,(shape[36][0],shape[36][1]),2,(0,0,255),-1)
		cv2.imshow("Output", image)
	    
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break
	except: # Just display the frames in case of any error
		_,image = cap.read() 
		image = cv2.flip(image,1)
		cv2.imshow('Output',image)
		k = cv2.waitKey(5) & 0xff
		if k == 27:
			break

cv2.destroyAllWindows()
cap.release()
