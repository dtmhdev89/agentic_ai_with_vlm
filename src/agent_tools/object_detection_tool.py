from typing import Annotated
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.chains.url_extractor import UrlExtractor
from src.models.yolo import Yolo


class ObjectDetectingAndCountingInput(BaseModel):
    text: str = Field(
        description="Path or URL to the image in the format PNG or JPG/JPEG"
    )


class ImageInput(BaseModel):
    image_path_or_url: str = Field(description="Image path or URL")


class ObjectDetectionTool:

    @staticmethod
    def _constuct_extractor_chain():
        """extractor_chain"""
        return UrlExtractor(
            model_name="qwen3:8b",
            llm_mode='ollama'
        ).extractor_chain
    
    @staticmethod
    def _construct_yolo_model():
        """yolo_model"""
        return Yolo()

    @tool(
        "detect_and_count_objects",
        description="Detect and count objects within the image. The return will be a dictionary, containing the counting dictionary (counting how many instance of each object class) and a list of dictionaries, containing the object names, confidence scores, and location in the image (in (x1, x2, y1, y2) format).",
        args_schema=ObjectDetectingAndCountingInput
    )
    @staticmethod
    def detect_and_count_objects(
        text: Annotated[str, "Path or URL to the image"]
    ):
        try:
            extractor_chain = ObjectDetectionTool._constuct_extractor_chain()
            parsed: ImageInput = extractor_chain.invoke({"input": text})
        except Exception as e:
            return f"Failed to extract image URL: {str(e)}"

        image_path_or_url = parsed.image_path_or_url
        if not image_path_or_url:
            return "No image URL found in the input."

        yolo_model = ObjectDetectionTool._construct_yolo_model()
        results = yolo_model.predict(image_path_or_url, verbose=False)

        detections = []
        counting: dict = {}

        # Process each result
        for result in results:
            boxes = result.boxes
            class_names = result.names

            for box in boxes:
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })

                counting[class_name] = counting.get(class_name, 0) + 1

        return str({'counting': counting, 'detections': detections})
