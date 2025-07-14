from ultralytics import YOLO
from PIL import Image

img = Image.open("/home/lenovo/gradio/UnionProject/test/bus.jpg")
detector = YOLO('weights/detector/model_11n.pt')
results = detector.predict(img)
results[0].show()