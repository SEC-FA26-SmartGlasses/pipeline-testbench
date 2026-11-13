from pipeline import Pipeline
from ultralytics import YOLO
## Change these if needed
from image_process import process_image
from ocr import img_to_text


## Tesseract OCR Pipeline ##
class TesseractPipeline(Pipeline):

    def __init__(
        self,
        model_path,
        config_oem_psm="--oem 3 --psm 6",
        padding=10,
        enable_skew=True,
        enable_clahe=True,
        enable_denoising=True,
        enable_threshold=True
    ):

        super().__init__()

        # Initialize YOLO model
        self.yolo_model = YOLO(model_path)

        # OCR settings
        self.config_oem_psm = config_oem_psm

        # Image processing settings
        self.padding = padding
        self.enable_skew = enable_skew
        self.enable_clahe = enable_clahe
        self.enable_denoising = enable_denoising
        self.enable_threshold = enable_threshold

    def run(self, image):

        # Process image
        boxes = process_image(
            image,
            self.yolo_model,
            padding=self.padding,
            enable_skew=self.enable_skew,
            enable_clahe=self.enable_clahe,
            enable_denoising=self.enable_denoising,
            enable_threshold=self.enable_threshold
        )

        # Extract text
        strings = img_to_text(
            boxes,
            config_oem=self.config_oem_psm
        )

        # Return full text
        return "\n".join(strings)