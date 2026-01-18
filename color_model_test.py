from ultralytics import YOLO26

model = YOLO26.from_pretrained("colour-science/colour-checker-detection-models")
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
model.predict(source="colortest1.jpg", save=True)

