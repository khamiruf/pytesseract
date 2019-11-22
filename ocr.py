from PIL import Image
import pandas as pd
import numpy as np
import pytesseract
import argparse
import cv2
import os


# construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image to be OCR'd") # path to image we're sending to the OCR system
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
    help="type of preprocessing to be done") # accepts two values: thresh (threshold) / blur
args = vars(ap.parse_args())


# load the image and convert it to grayscale (binarize it) & write it to disk
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# check to see if we should apply thresholding to preprocess the image
if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    

# make a check to see if median blurring should be done to remove noise
elif args["preprocess"] == "blur":
    gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temp file so we can apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)


# load the image as a PIL/Pillow image, apply OCR, and then delete the temp file
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
try_config = r'-c preserve_interword_spaces=1x1 --psm 5 --oem 3'
custom_oem_psm_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(Image.open(filename), lang="chi_sim", config=custom_oem_psm_config)
os.remove(filename)

outfile_txt = "output/output_text.txt"

with open(outfile_txt, "w+", encoding="utf-8-sig") as f:
    f.writelines(text)


# tried for multiple columns to split into data format

# with open(outfile_txt, "r", encoding="utf-8-sig") as f:
#     lines = f.readlines()

# # remove spaces
# lines = [','.join(line.split()) for line in lines]

# # finally, write lines in the file
# with open('out_text.txt', 'w', encoding='utf-8-sig') as f:
#     f.writelines(s + '\n' for s in lines)