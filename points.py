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
from scipy.ndimage import gaussian_filter

# The function is used for calculation of MAR for the mouth
def MAR(point1,point2,point3,point4,point5,point6,point7,point8):
	mar = (dst(point1,point2) + dst(point3,point4) + dst(point5,point6))/(3.0*dst(point7,point8))
	return mar

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
scroll = np.array([])
eyeopen = np.array([])
t1 = np.array([])
t2 = np.array([])
t3 = np.array([])
t4 = np.array([])
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
while(time.time() - currenttime <= 25): #The calibration code will run for 23 seconds.
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
		mar = MAR(shape[50],shape[58],shape[51],shape[57],shape[52],shape[56],shape[48],shape[54])
		EARdiff = (lefteye - righteye)*100
		elapsedTime = time.time() - currenttime
		if elapsedTime < 5.0: # recording the phase where both eyes are open
			cv2.putText(image,'Keep Both Eyes Open',(0,100), font, 1,(0,0,0),2,cv2.LINE_AA)
			eyeopen = np.append(eyeopen,[EARdiff])
			t1 = np.append(t1,[elapsedTime])
		elif elapsedTime > 5.0 and elapsedTime < 10.0: # recording the phase when only left eye is closed
			cv2.putText(image,'Close Left Eye',(0,100), font, 1,(0,0,0),2,cv2.LINE_AA)
			lclick = np.append(lclick,[EARdiff])
			t2 = np.append(t2,[elapsedTime])
		elif elapsedTime > 12.0 and elapsedTime < 17.0: # recording the phase for only right eye
			cv2.putText(image,'Open Left eye and close Right Eye',(0,100), font, 1,(0,0,0),2,cv2.LINE_AA)
			rclick = np.append(rclick,[EARdiff])
			t3 = np.append(t3,[elapsedTime])
		elif elapsedTime > 19.0 and elapsedTime < 24.0: # recording the phase for open mouth
			cv2.putText(image,'Open Your Mouth',(0,100),font,1,(0,0,0),2,cv2.LINE_AA)
			scroll = np.append(scroll,[mar])
			t4 = np.append(t4,[elapsedTime])
		else: # no recording done in this phase. It is just a small lag
			pass 
		for (x,y) in shape: # prints facial landmarks on the face
			cv2.circle(image,(x,y),2,(0,255,0),-1)
		cv2.imshow('image',image) # display of image
	if cv2.waitKey(5) & 0xff == 27: 
		break

# Plotting the recorded points when both eyes are open
plt.subplot(2,2,1)
eyeopen_smooth = gaussian_filter(eyeopen,sigma = 5)
plt.title('Both Eyes Open')
plt.plot(t1,eyeopen_smooth)

# Plotting the recorded points when left eye is closed
plt.subplot(2,2,2)
lclick_smooth = gaussian_filter(lclick,sigma = 5)
plt.title('Left click')
plt.plot(t2,lclick_smooth)

# Plotting the recorded points when right eye is closed
plt.subplot(2,2,3)
rclick_smooth = gaussian_filter(rclick,sigma = 5)
plt.title('Right click')
plt.plot(t3,rclick_smooth)

# Plotting the recorded points when the mouth is opened
plt.subplot(2,2,4)
scroll_smooth = gaussian_filter(scroll,sigma = 5)
plt.title('Scroll Mode')
plt.plot(t4,scroll_smooth)

plt.show() # Display of graph. Press any key to exit the graph
cap.release()
cv2.destroyAllWindows()


# The second snippet of code consists of the main code, where all the cursor controlling using face gestures will take place
cap = cv2.VideoCapture(0)
MARlist = np.array([]) # Initialization of a MAR list which will be resued after every 30 iterations
scroll_status = 0 # Checks the scroll status: 1:ON 0:OFF
openeyes = np.mean(eyeopen) # Calculates mean of the recorded values for the case when both eyes are opened.
leftclick = np.average(lclick) - 1.5# Calculates mean of the recorded values for left click. Subtracting a constant to make it a bit more flexible
rightclick = np.average(rclick) + 1.5 # Calculates mean of the recorded values for right click. Adding a constant to make it a bit more flexible
scrolling = np.mean(scroll)
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
			mar = MAR(shape[50],shape[58],shape[51],shape[57],shape[52],shape[56],shape[48],shape[54]) # Calculation of the MAR of mouth
			if EARdiff < leftclick: # Left click will be initiated if the EARdiff is less than the leftclick calculated during calibration
				pag.click(button = 'left') 
			elif EARdiff > rightclick: # Right click will be initiated if the EARdiff is more than the rightclick calculated during calibration
				pag.click(button = 'right')
				
					
			cv2.line(image,(250,250),(h,k),(0,0,0),1) # Draws a line between the tip of the nose and the centre of reference circle
			# Controls the move of mouse according to the position of the found nose.
		MARlist = np.append(MARlist,[mar]) # Appending the list at every iteration
		if len(MARlist) == 30: # till it reaches a size of 30 elements
			mar_avg = np.mean(MARlist)
			MARlist = np.array([]) # Resetting the MAR list
			if int(mar_avg*100) > int(scrolling*100):
				if scroll_status == 0:
					scroll_status = 1
				else:
					scroll_status = 0
		if scroll_status == 0:
			if((h-250)**2 + (k-250)**2 - 50**2 > 0):
				a = angle(shape[33]) # Calculates the angle
				if h > 250: # The below conditions set the conditions for the mouse to move and that too in any direction we desire it to move to.
					pag.moveTo(pag.position()[0]+(10*m.cos(-1.0*a)),pag.position()[1]+(10*m.sin(-1.0*a)),duration = 0.01)
				else:
					pag.moveTo(pag.position()[0]-(10*m.cos(-1.0*a)),pag.position()[1]-(10*m.sin(-1.0*a)),duration = 0.01)
		else: #Enabling scroll status
			cv2.putText(image,'Scroll mode ON',(0,100),font,1,(0,0,0),2,cv2.LINE_AA)
			if k > 300: 
				pag.scroll(-1)
			elif k < 200:
				pag.scroll(1)
			else:
				pass
		
	    	
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
