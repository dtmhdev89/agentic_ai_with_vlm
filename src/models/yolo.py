from ultralytics import YOLO


class Yolo:
    """YOLO model"""

    def __new__(cls, model_name="yolo11x.pt"):
        return YOLO(model_name)
