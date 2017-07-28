# import the necessary packages
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
import argparse
import cv2
import os
import imutils

print('Import successful')

# load image, resize as desired, and convert to grayscale
image = cv2.imread("sample_01.jpg")
# image = imutils.resize(image, height = 200)
cv2.imshow("Original", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

# threshold image
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Threshold", gray)

# blur image to decrease noise
gray = cv2.medianBlur(gray, 3)
cv2.imshow("Blurred", gray)

# create temporary image to run OCR
imagename = "{}.png".format(os.getpid())
cv2.imwrite(imagename, gray)
binary_image = Image.open(imagename)

# identify text in image and print to console
text = pytesseract.image_to_string(binary_image)
print(text)

# remove image from folder (optional)
os.remove(imagename)

# create .txt file and store data
f = open('output.txt', 'w')
f.write(text + '\n')
f.close()

# confirm code ran as intended and wait for user to press any key
print('Good job - everything worked')
cv2.waitKey(0)






