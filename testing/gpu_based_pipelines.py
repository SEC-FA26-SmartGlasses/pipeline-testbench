from testing.dataset import load_dataset
from testing.pipeline_def import Pipeline
from testing.score import score_case, evaluate_pipeline

from paddleocr import PaddleOCR
from paddleocr import PPStructureV3
from paddleocr import PaddleOCRVL

from paddlex import create_pipeline
# import keras_ocr # keras-ocr will automatically download pretrained weights for the detector and recognizer.
import torch
from transformers import AutoModel, AutoTokenizer

class PaddlePipeline(Pipeline):

    def __init__(self, det_model, rec_model):
        super().__init__()
        self.ocr = PaddleOCR(
            use_textline_orientation=True, use_doc_unwarping=True, lang="en",
            text_recognition_model_name=rec_model,
            text_detection_model_name=det_model,
        )

    def run(self, image):
        output = self.ocr.predict(image)
        output = "\n".join(output[0]["rec_texts"])
        return output


# paddle for gpu
class PaddleVLPipeline(Pipeline):
    def __init__(self):
        # straight paddle-vl 
        self.ocr = PaddleOCRVL(pipeline_version="v1.6")
  
    def run(self, image):
        output = self.ocr.predict(image)
        return "\n".join(res.markdown["markdown_texts"] for res in output)

        
# paddle w/ doc layout module  
class PaddleStructureV3Pipeline(Pipeline):
    def __init__(self):
        self.ocr = PPStructureV3(
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            text_recognition_model_name="PP-OCRv5_server_rec",
            text_detection_model_name="PP-OCRv5_server_det",
        )


    def run(self, image):
        output = self.ocr.predict(
            input=image, 
            
        )
        return "\n".join(res.markdown["markdown_texts"] for res in output)

class UnlimitedOCRPipeline(Pipeline):
    model_name = "baidu/Unlimited-OCR-2"
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                use_safetensors=True,
                torch_dtype=torch.bfloat16
                )
        self.ocr = self.model.eval().cuda() #!
         
    def run(self, image):
        output = self.ocr.infer(
          self.tokenizer,
          prompt='<image>document parsing.',
          image_file = image,
          output_path='out/',
          crop_mode=True,
          no_repeat_ngram_size=35, ngram_window=128,
          save_results=False,
        )
        return output 