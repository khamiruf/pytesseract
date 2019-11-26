from PIL import Image
import pandas as pd
import numpy as np
import pytesseract
import argparse
import cv2
import os

def load_image(img, preprocess = "thresh"):
    # load the image and convert it to grayscale (binarize it) & write it to disk
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check to see if we should apply thresholding to preprocess the image
    if preprocess == "thresh":
        gray = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # make a check to see if median blurring should be done to remove noise
    elif preprocess == "blur":
        gray = cv2.medianBlur(gray, 3)

    return gray

def write_temp_file(gray):
    # write the grayscale image to disk as a temp file so we can apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    return filename

def apply_ocr(filename):
        
    # load the image as a PIL/Pillow image, apply OCR, and then delete the temp file
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    try_config = r'-c preserve_interword_spaces=1x1 --psm 5 --oem 3'
    custom_oem_psm_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(Image.open(filename), lang="chi_sim", config=custom_oem_psm_config)
    os.remove(filename)

    return text


if __name__ == "__main__":
    try:
        # get number of files in images dir
        num_of_files = len([name for name in os.listdir('./images')])
        count = 0

        for entry in os.scandir('./images'):
            if count < num_of_files:
                gray = load_image(entry.path)
                filename = write_temp_file(gray)
                output = apply_ocr(filename)
                outfile_txt = "output/output_text_{}.txt".format(count+1)

                with open(outfile_txt, "w+", encoding="utf-8-sig") as f:
                    f.writelines(output)

                print(count)
                count += 1

    except KeyboardInterrupt:
        pass