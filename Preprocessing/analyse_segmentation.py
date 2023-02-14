import cv2

import numpy as np
from matplotlib import pyplot as plt
import commons as cp

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

world_data=r'data_unzipped\data\002'
cap = cv2.VideoCapture(world_data+r'\world.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Select frame to select template
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    cv2.imshow('Frame',frame)
    # plt.imshow(frame, extent=[-1, 1, -1, 1])
    # plt.show()
    # Press Q on keyboard to  exit
    if cv2.waitKey(50) & 0xFF == ord('p'):
      break
 
  # Break the loop
  else: 
    if cv2.waitKey(50) & 0xFF == ord('q'):
      break

# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
roi1=cv2.selectROI('Window_name', img_gray)

cropped_im=cp.crop_image(roi1,img_gray)
plt.imshow(cropped_im, extent=[-1, 1, -1, 1])
plt.show()

fig,ax=plt.subplots(1,1)
detected=cp.image_correlation(frame,img_gray,cropped_im)
ax.imshow(detected,extent=[-1, 1, -1, 1])


#Continue the loop for all frames and all borders
