import json
import evaluate
from testing.dataset import load_dataset
from testing.pipeline_def import Pipeline


bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
cer = evaluate.load("cer")
wer = evaluate.load("wer")

def score_case(ground_truth, prediction):
    scores = {}
    
    scores["bleu"] = bleu.compute(predictions=[prediction], references=[ground_truth])["bleu"]
    scores["rougeL"] = float(rouge.compute(predictions=[prediction], references=[ground_truth])["rougeL"])
    scores["cer"] = float(cer.compute(predictions=[prediction], references=[ground_truth]))
    scores["wer"] = float(wer.compute(predictions=[prediction], references=[ground_truth]))

    return scores




def evaluate_pipeline(cases, pipeline):
    all_scores = []
    for case in cases:
        prediction = pipeline.run(case["image"])
        scores = score_case(case["ground_truth"], prediction)
        scores["prediction"] = prediction
        all_scores.append(scores)
    return all_scores