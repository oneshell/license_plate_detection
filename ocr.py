# Steve Mitchell
# UMD, College Park

# Sign & License Plate Detection

# Workflow:
# 1. Read image
# 2. Grayscale, blur, & edge detection
# 3. Use detected edges to identify region of image containing text
# 4. Implement OCR to identify text

####################################################

# Import required packages
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
import cv2
import os
import imutils
import numpy as np

print('Import successful' + '\n')

####################################################

# Initialize variables describing anticipated sign type
# will use these descriptions for number of contour points
stop_sign = 8
license_plate = 4
speed_sign = 4
outlet_sign = 4
yield_sign = 3

####################################################

# Load image, resize as desired, and convert to grayscale
image = cv2.imread("speed_limit_04.jpg")
# Image = imutils.resize(image, height = 200)
cv2.imshow("Original", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

# Blur image to decrease noise
gray = cv2.medianBlur(gray, 3)
cv2.imshow("Blurred", gray)

# Find edges
edged = cv2.Canny(gray, 30, 200)
cv2.imshow("Edged", edged)

####################################################

# Find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

# Loop over our contours
for c in cnts:
    # Approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # If our approximated contour has x-points, then
    # we can assume that we have found our screen
    if len(approx) == outlet_sign:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Countours", image)

####################################################

# create polygon (trapezoid) mask to select region of interest
mask = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
pts = np.array([screenCnt], dtype=np.int32)
print(len(screenCnt))
print(screenCnt)
print(screenCnt.shape)
# cv2.fillConvexPoly(mask, pts, 255)
# cv2.imshow("Mask", mask)
#
# # apply mask and show masked image on screen
# masked = cv2.bitwise_and(image, image, mask=mask)
# cv2.imshow("Region of Interest", masked)

####################################################

# Threshold image
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Threshold", gray)

# Invert image if desired
gray = cv2.bitwise_not(gray)
cv2.imshow("Flipped", gray)

# Create temporary image to run OCR
imagename = "{}.png".format(os.getpid())
cv2.imwrite(imagename, gray)
binary_image = Image.open(imagename)

# Identify text in image and print to console
text = pytesseract.image_to_string(binary_image)
print(text)

# Remove image from folder (optional)
os.remove(imagename)

####################################################

# Create .txt file and store data
f = open('output.txt', 'w')
f.write(text + '\n')
f.close()

####################################################

# Confirm code ran to complettion & wait for user to press any key
print('\n' + 'Good job - everything worked')
cv2.waitKey(0)






