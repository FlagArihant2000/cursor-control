# Summer-Project
This repository consists of the summer project 2019 along with some side projects done during the month of June

VDP.py - This project basically deals with making a virtual drawing pad and drawing various shapes without the use of mouse. The video frames are converted to HSV and is thresholded to filter only the green colour, which will act as the "stylus" for us to draw with. A contour is drawn around the "stylus" and then a centroid is calculated for it, which acts as the point using which drawing will take place. The drawings are printed corresponding to the coordinates of the centroid on the real video on another white window. The color and the size of the paint can be changed using the trackbars provided with the code.
