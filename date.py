import cv2
import pytesseract
import pandas as pd
import os
import re

# If using Windows, set the path to Tesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Path to the folder containing images
image_folder = "yolo_new1"

# Output CSV file
csv_file = "extracted_dates.csv"

# Regular expression to match different date formats (DD-MM-YYYY, MM/DD/YYYY, YYYY-MM-DD, etc.)
date_pattern = r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"

# Initialize a list to store extracted dates
extracted_dates = []

# Process each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # Check for image files
        image_path = os.path.join(image_folder, filename)

        # Read the image
        image = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply OCR
        extracted_text = pytesseract.image_to_string(gray)

        # Find date using regex
        matches = re.findall(date_pattern, extracted_text)

        # If a date is found, store the first valid one
        if matches:
            extracted_dates.append([matches[0]])

# Convert to DataFrame and save as CSV
df = pd.DataFrame(extracted_dates, columns=["Date"])
df.to_csv(csv_file, index=False, header=False)

print(f"Dates extracted from images and saved to {csv_file}")
