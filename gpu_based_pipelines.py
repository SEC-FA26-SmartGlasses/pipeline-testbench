from testing.dataset import load_dataset
from testing.pipeline_def import Pipeline
from testing.score import score_case, evaluate_pipeline

from paddleocr import PaddleOCR
from paddleocr import PPStructureV3
from paddleocr import PaddleOCRVL
# import keras_ocr # keras-ocr will automatically download pretrained weights for the detector and recognizer.
import torch
from transformers import AutoModel, AutoTokenizer

class PaddlePipeline(Pipeline):

    def __init__(self):
        super().__init__()
        self.ocr = PaddleOCR(
            use_textline_orientation=True, use_doc_unwarping=True, lang="en"
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
        print(output)
        output = "\n".join(output[0]["rec_texts"])
        return output

        
# paddle w/ doc layout module  
class PaddleStructureV3Pipeline(Pipeline):
    def __init__(self):
        self.ocr = PPStructureV3(
            lang="en", 
            use_doc_orientation_classify=True, 
            use_doc_unwarping=True, 
            use_textline_orientation=True
        )
  
    def run(self, image):
        output = self.ocr.predict(image)
        output = "\n".join(output[0]["rec_texts"])
        return output

# # keras
# class KerasOCRPipeline(Pipeline):
#     def __init__(self):
#         self.ocr = keras_ocr.pipeline.Pipeline()
         
#     def run(self, image):
#         output = self.ocr.recognize(image) # returns tuple of (word, box)
#         output = "\n".join([word for word, box in output[0]]) 
#         return output

# unlimited (could be good for 'hard' multi col)
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