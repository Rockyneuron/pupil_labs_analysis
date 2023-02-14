import cv2
import numpy as np

def crop_image(roi,im):
  """Given a tupple of 4 elements crop an image respect
  to a given region of interest

  Args:
      roi (tupple): () 
  """
  cropped_image = im[int(roi[1]):int(roi[1]+roi[3]), 
                      int(roi[0]):int(roi[0]+roi[2])]
  return cropped_image

def image_correlation(img_rgb,img_gray,template):
  """Check image correlation performing a convolution

  Args:img_gray
      im1 (_type_): _description_
      template (_type_): _description_
  """
  # Store width and height of template in w and h
  w, h = template.shape[::-1]
    
  # Perform match operations.
  res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    
  # Specify a threshold
  threshold = 0.8
    
  # Store the coordinates of matched area in a numpy array
  loc = np.where(res >= threshold)
    
  # Draw a rectangle around the matched region.
  for pt in zip(*loc[::-1]):
      cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
    
  # Show the final image with the matched area.
  return img_rgb
