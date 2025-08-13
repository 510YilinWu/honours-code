import cv2
import numpy as np
import os
from datetime import datetime

""" 
Connect to two webcams and capture a single frame from each after `space` is pressed.
Continue capturing frame pairs after each press of `space`.
Pressing 'q' will stop the capture
"""

def capture_frame(path,fileBase):
    print("Connecting to webcams...")
    cap0 = cv2.VideoCapture(0)
    if not cap0.isOpened():
        print("Error: Could not open webcam 0.")
        return  
    else:
        print("Webcam 0 opened successfully.")
    focus = 0  # min: 0, max: 255, increment:5
    cap0.set(cv2.CAP_PROP_AUTO_WB, 0) # disable auto whitebalance
    focus = 0  # min: 0, max: 255, increment:5
    cap0.set(28, focus)
    cap0.set(cv2.CAP_PROP_TEMPERATURE, 5000) # Set a specific color temperature

    cap1 = cv2.VideoCapture(1)
    if not cap1.isOpened():
        print("Error: Could not open webcam 1.")
        return
    else:
        print("Webcam 1 opened successfully. \n Press 'space' to capture frames, 'q' to quit.")
    focus = 0  # min: 0, max: 255, increment:5
    cap1.set(cv2.CAP_PROP_AUTO_WB, 0) # disable auto whitebalance
    focus = 0  # min: 0, max: 255, increment:5
    cap1.set(28, focus)
    cap1.set(cv2.CAP_PROP_TEMPERATURE, 5000) # Set a specific color temperature

    image_count = 0  # Counter for the number of images captured


    while image_count < 4:  # Stop after capturing 2 images

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            print("Error: Could not read frame.")
            break
        
        if not os.path.exists(path):
            os.makedirs(path)
        # Find the next available file number
        existing_files = [f for f in os.listdir(path) if f.startswith(fileBase) and f.endswith('.png')]
        file_numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if '_' in f]
        next_file_number = max(file_numbers, default=-1) + 1
        fileBaseWithNumber0 = f"{fileBase}0_{next_file_number:02d}.png"
        fileBaseWithNumber1 = f"{fileBase}1_{next_file_number:02d}.png"
        # Save the frame
        fName0 = os.path.join(path, fileBaseWithNumber0)
        cv2.imwrite(fName0, frame0)
        fName1 = os.path.join(path, fileBaseWithNumber1)
        cv2.imwrite(fName1, frame1)
        print(f"Frames captured and saved as '{fName0}'")

        image_count += 2  # Increment the counter

    cap0.release()
    cap1.release()

if __name__ == "__main__":
    # Define the path in format \YYYY\MM\DD\
    now = datetime.now()
    date = now.strftime("%Y/%m/%d")
    path = os.path.join('/Users/yilinwu/Desktop/honours data/', 'bbt', date)
    fileBase = 'cam'
    capture_frame(path,fileBase)
