from testing.dataset import load_dataset
from testing.pipeline_def import Pipeline
from testing.score import evaluate_pipeline

from paddleocr import PaddleOCR

from pathlib import Path

from gpu_based_pipelines import PaddlePipeline, PaddleVLPipeline, PaddleStructureV3Pipeline, UnlimitedOCRPipeline


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


pipelines = [PaddleStructureV3Pipeline, UnlimitedOCRPipeline] # defer initialization

for pipeline_class in pipelines:
    pipeline = pipeline_class()
    output_dir = Path("results") / type(pipeline).__name__
    all_scores = evaluate_pipeline(cases, pipeline, output_dir=output_dir)
