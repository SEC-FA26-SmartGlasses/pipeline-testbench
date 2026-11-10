from testing.dataset import load_dataset
from testing.pipeline_def import Pipeline
from testing.score import score_case, evaluate_pipeline

from paddleocr import PaddleOCR







cases = load_dataset("E:/test_dataset")


class OCRPipeline(Pipeline):

    def __init__(self):
        super().__init__()
        self.ocr = PaddleOCR(
            use_textline_orientation=True, use_doc_unwarping=True, lang="en"
        )

    def run(self, image):
        output = self.ocr.predict(image)
        output = "\n".join(output[0]["rec_texts"])
        return output


pipeline = OCRPipeline()



all_scores = evaluate_pipeline(cases, pipeline)

# write to json
import json

with open("all_scores.json", "w") as f:
    json.dump(all_scores, f, indent=4)
