from ultralytics import YOLO
import cv2
# Load a pretrained YOLOv8n model
model = YOLO(model='runs/detect/train/weights/best.pt')

img = cv2.imread('datasets/val/images/maksssksksss0.png')
# Run inference on 'bus.jpg' with arguments
result = model.predict(img, save=True)
print(result[0])