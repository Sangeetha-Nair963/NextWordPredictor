import io
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='./cred.json'
from google.cloud import vision
import re
from tkinter import *    
from tkinter import filedialog as fd


def openFile():  
    # selecting the file using the askopenfilename() method of filedialog 
    f_types = [("files", [".png",'.jpeg','jpg'])] 
    the_file = fd.askopenfilename(  
    title = "Select an image to upload",  
    filetypes = f_types
    )
    return the_file
   
def detect_handwritten_ocr(path):
    """Detects handwritten characters in a local image.
    Args:
    path: The path to the local file.
    """
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Language hint codes for handwritten OCR:
    # en-t-i0-handwrit, mul-Latn-t-i0-handwrit
    # Note: Use only one language hint code per request for handwritten OCR.
    image_context = vision.ImageContext(
        language_hints=['en-t-i0-handwrit'])

    response = client.document_text_detection(image=image,
                                              image_context=image_context)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    

    return('{}'.format(response.full_text_annotation.text))
    
