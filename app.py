import streamlit as st
import pandas as pd
import cv2, os, time
import numpy as np
from ultralytics import YOLO

def load_image(image):
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image

def save_image(filename, np_image, string):
    cv2.imwrite(os.path.join("outputs", string + filename), np_image)

def model_predict(model, np_image):
    result = model.predict(np_image, save=True)
    return result

def read_image():
    # time.sleep(10)
    size = len(os.listdir("runs/detect/")) - 1
    print(size)
    if size == 1:
        predict_image_path = "predict"
    else:
        predict_image_path = f"predict{size}"
    image_path = os.path.join("runs/detect/", predict_image_path, "image0.jpg")
    print("Image path", image_path)
    return cv2.imread(image_path)

# Load a pretrained YOLOv8n model
model = YOLO(model='runs/detect/train/weights/best.pt')

st.write("""
# Face Mask Detection Using Yolo
In this app upload the Images of the peoples app will helps us to say who have Mask, who have No_Mask, who have Partial_Mask
""")

images = st.file_uploader("Face Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

for image in images:

    #Loading the images
    opencv_image = load_image(image)
    #save input images to the folder
    save_image(image.name, opencv_image, string="Input")
    st.image(opencv_image, channels="BGR")
    #predict the data
    model.predict(opencv_image, save=True)
    #read the predicted data
    result_image = read_image()
    st.image(result_image, channels="BGR")
    #save the predicted images
    save_image(image.name, result_image, string="Output")
    
