from ultralytics import YOLO

model = YOLO('runs/detect/train3/weights/last.pt')

results = model(source='test/images',conf=0.7, save=True)