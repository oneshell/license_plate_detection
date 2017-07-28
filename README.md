# license_plate_detection

Code uses pytesseract, opencv, and imutils to read in an image file, identify text in the image, and print the interpreted text to the console.

You'll need to install Tesseract on your machine.  Recommend: https://github.com/UB-Mannheim/tesseract/wiki

Once you import pytesseract, tell Python where Tesseract is located:

import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'






