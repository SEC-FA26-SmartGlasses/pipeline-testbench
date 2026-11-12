from pipeline import Pipeline
from ultralytics import YOLO
from src.text_extraction.image_process import process_image
from src.ocr.ocr import img_to_text


## Image Processing ##
class ImageProcessing:

    # Enable different filters
    def __init__(
        self,
        model_path,
        padding=10,
        enable_skew=True,
        enable_clahe=True,
        enable_denoising=True,
        enable_threshold=True
    ):

        self.yolo_model = YOLO(model_path)

        # Image processing settings
        self.padding = padding
        self.enable_skew = enable_skew
        self.enable_clahe = enable_clahe
        self.enable_denoising = enable_denoising
        self.enable_threshold = enable_threshold

    def run(self, image):

        return process_image(
            image,
            self.yolo_model,
            padding=self.padding,
            enable_skew=self.enable_skew,
            enable_clahe=self.enable_clahe,
            enable_denoising=self.enable_denoising,
            enable_threshold=self.enable_threshold
        )


## Tesseract OCR Pipeline ##
class TesseractPipeline(Pipeline):

    def __init__(
        self,
        image_processor,
        config_oem_psm="--oem 3 --psm 6"
    ):

        super().__init__()

        self.image_processor = image_processor
        self.config_oem_psm = config_oem_psm

    def run(self, image):

        # Process image
        boxes = self.image_processor.run(image)

        # Extract text
        strings = img_to_text(
            boxes,
            config=self.config_oem_psm
        )

        # Return full text
        return "\n".join(strings)