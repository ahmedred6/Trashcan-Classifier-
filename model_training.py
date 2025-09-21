from ultralytics import YOLO
model = YOLO("yolov8n-cls.pt")
model.train(
    data="garbage_split",
    epochs=50,
    imgsz=224,
    augment=True
)
model.val()