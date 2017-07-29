# Steve Mitchell
# UMD, College Park

# Simple OCR detection

####################################################

# Import required packages
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
import cv2
import os
import imutils

print('Import successful!' + '\n')

####################################################

# Load image, resize as desired, and convert to grayscale
original = cv2.imread("sample_01.jpg")
# original = imutils.resize(original, height = 200)
cv2.imshow("Original Image", original)
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)

# Blur image to decrease noise
blurred = cv2.medianBlur(gray, 3)
cv2.imshow("Blurred", blurred)

# Threshold image
thres = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Threshold", thres)

# Flip (invert) image if desired
# flipped = cv2.bitwise_not(thres)
# cv2.imshow("Flipped", flipped)

####################################################

# Create temporary image to run OCR
imagename = "{}.png".format(os.getpid())
cv2.imwrite(imagename, thres)
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

