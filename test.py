from ultralytics import YOLO
import cv2

model = YOLO("combine.pt")

img = cv2.imread("test.jpg")

results = model(img)

print(results)