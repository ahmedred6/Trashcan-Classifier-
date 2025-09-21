from ultralytics import YOLO
trained_model = YOLO("runs/classify/train/weights/best.pt")
metrics = trained_model.val(split="test")
print(metrics)