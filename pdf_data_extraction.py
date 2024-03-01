"""
This script is designed to extract text from a PDF file by converting each page of the PDF into an image and then using OCR (Optical Character Recognition) to read the text from these images. It utilizes the `pdf2image` library to convert PDF pages to images and the `pytesseract` library, a Python wrapper for Google's Tesseract-OCR Engine, to perform the OCR process.
"""

from pdf2image import convert_from_path
import numpy as np
import pytesseract


def extract_data(pdf_path):
    """
    Extracts text from a PDF file by converting each page to an image and then using OCR to read the text.

    Parameters:
    - pdf_path (str): The file path of the PDF from which to extract text.

    Returns:
    - str: A string containing all the extracted text from the PDF.
    """
    # Initialize the variable to store the extracted text
    extracted_text = ""

    # Converts the PDF to images
    pages = convert_from_path(pdf_path, dpi=300)

    # Loop through the images and extract text
    for page in pages:
        # Use Tesseract to extract text from the image
        text = pytesseract.image_to_string(np.array(page))

        # Append the text to the extracted_text variable
        extracted_text += text

    # Return the extracted text
    return extracted_text
