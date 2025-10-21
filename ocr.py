import cv2
import pytesseract
import pandas as pd
import os
import re



# Path to the folder containing images
image_folder = "yolo_new1"

# Output CSV file
csv_file = "extracted_dates_times.csv"

# Regular expression to match date (DD-MM-YYYY, MM/DD/YYYY, YYYY-MM-DD, etc.)
date_pattern = r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"

# Regular expression to match time (supports 10.3.5, 10:3:5, 10:30 AM, but NOT 10-3-5)
time_pattern = r"\b(\d{1,2}[:.]\d{1,2}[:.]\d{1,2}|\d{1,2}[:.]\d{1,2}(?:\s?(?:AM|PM|am|pm))?)\b"

# Initialize a list to store extracted data
extracted_data = []

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

        # Find date and time using regex
        date_match = re.findall(date_pattern, extracted_text)
        time_match = re.findall(time_pattern, extracted_text)

        # Get the first valid date and time
        date_part = date_match[0] if date_match else "N/A"
        time_part = time_match[0] if time_match else "N/A"

        # Append extracted values
        extracted_data.append([date_part, time_part])

# Convert to DataFrame and save as CSV
df = pd.DataFrame(extracted_data, columns=["Date", "Time"])
df.to_csv(csv_file, index=False, header=False)

print(f"Date and time extracted from images and saved to {csv_file}")
