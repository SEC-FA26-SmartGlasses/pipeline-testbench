from testing.dataset import load_dataset
from testing.pipeline_def import Pipeline
from testing.tesseract_pipelines import TesseractPipeline
from testing.score import evaluate_pipeline, rescore_results
from testing.postprocess_text import clean_ocr_text

from paddleocr import PaddleOCR

from pathlib import Path

from testing.gpu_based_pipelines import PaddlePipeline, PaddleVLPipeline, PaddleStructureV3Pipeline, UnlimitedOCRPipeline


cases = load_dataset("E:/test_dataset")

pipelines = [TesseractPipeline] # defer initialization

# for pipeline_class in pipelines:
#     if pipeline_class is TesseractPipeline:
#         pipeline = TesseractPipeline(
#             "text_extraction_model.pt",
#         )
#     else:    
#         pipeline = pipeline_class()
#     output_dir = Path("results_11_12_2026_TESSERACT/raw") / type(pipeline).__name__
#     all_scores = evaluate_pipeline(cases, pipeline, output_dir=output_dir)


rescore_results(clean_ocr_text, "results_11_12_2026_TESSERACT/raw", "results_11_12_2026_TESSERACT/postprocessed")