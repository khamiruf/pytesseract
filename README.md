# OCR for Transcribing Chinese Images to Text

### Configurations: 
- run ```pip install -r requirements.txt``` before anything (only once)
- from https://stackoverflow.com/a/53672281:
```
1. Install tesseract using windows installer available at: https://github.com/UB-Mannheim/tesseract/wiki

2. Note the tesseract path from the installation.Default installation path at the time the time of this edit was: C:\Users\USER\AppData\Local\Tesseract-OCR. It may change so please check the installation path.

3. pip install pytesseract

4. set the tesseract path in the script before calling image_to_string:

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
```

### Running of script:

1. Download image to /images folder
2. Run script by "python ocr.py"

Example:
```
python ocr.py
```
3. Output will be generated into `/output` folder
