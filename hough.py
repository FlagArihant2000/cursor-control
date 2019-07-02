import numpy as np
import cv2

def hough_line(img):
	thetas = np.deg2rad(np.arange(-90.0,90.0))
	width,height = img.shape
	diag_len = int(np.ceil(np.sqrt(width*width + height*height)))
	rhos = np.linspace(-diag_len,diag_len,diag_len*2.0)
	
	cos_t = np.cos(thetas)
	sin_t = np.sin(thetas)
	num_thetas = len(thetas)
	print(num_thetas,diag_len)
	accumulator = np.zeros((2*diag_len,num_thetas),dtype = np.uint64)
	y_idxs,x_idxs = np.nonzero(img)
	for i in range(len(x_idxs)):
		x = x_idxs[i]
		y = y_idxs[i]
	
		for t_idx in range(num_thetas):
			rho = int(round(x*cos_t[t_idx] + y*sin_t[t_idx])) + diag_len
			accumulator[rho,t_idx]+=1
	return accumulator,thetas,rhos

image = np.zeros((640,480))
image[10:110,10:110] = np.eye(100)
accumulator,thetas,rhos = hough_line(image)

idx = np.argmax(accumulator)
rho = rhos[int(idx/accumulator.shape[1])]
theta = thetas[idx % accumulator.shape[1]]
print(rho,np.rad2deg(theta))

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
