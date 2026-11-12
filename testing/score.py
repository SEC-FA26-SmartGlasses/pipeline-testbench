import json
from pathlib import Path

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


def _result_path(output_path, case, test_index):
    difficulty = case.get("difficulty")
    label = case.get("label")
    variant = case.get("variant")
    if all(isinstance(value, str) and value for value in (difficulty, label, variant)):
        recipe_prefix = label.replace("recipe", "r", 1)
        return output_path / difficulty / label / f"{recipe_prefix}_{variant}.json"

    return output_path / f"test_{test_index:04d}.json"


def evaluate_pipeline(cases, pipeline, output_dir=None):
    all_scores = []
    metric_names = ("bleu", "rougeL", "cer", "wer")
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for test_index, case in enumerate(cases, start=1):
        prediction = pipeline.run(case["image"])
        scores = score_case(case["ground_truth"], prediction)
        result = {
            **scores,
            "prediction": prediction,
            "ground_truth": case["ground_truth"],
        }
        for field in ("difficulty", "label", "variant"):
            if field in case:
                result[field] = case[field]

        all_scores.append(result)
        if output_path is not None:
            result_path = _result_path(output_path, case, test_index)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.open("w", encoding="utf-8") as result_file:
                json.dump(result, result_file, indent=2, ensure_ascii=False)
                result_file.write("\n")

    if output_path is not None:
        def average_scores(results):
            return {
                metric: (
                    sum(result[metric] for result in results) / len(results)
                    if results
                    else None
                )
                for metric in metric_names
            }

        def average_by_variant(results):
            averages = {"overall": average_scores(results)}
            variants = sorted({
                result["variant"].casefold()
                for result in results
                if isinstance(result.get("variant"), str)
            })
            for variant in variants:
                variant_scores = [
                    result for result in results
                    if isinstance(result.get("variant"), str)
                    and result["variant"].casefold() == variant
                ]
                averages[variant] = average_scores(variant_scores)
            return {"average": averages}

        summary_by_difficulty = {}
        difficulties = sorted({
            result["difficulty"].casefold()
            for result in all_scores
            if isinstance(result.get("difficulty"), str)
        })
        for difficulty in difficulties:
            difficulty_scores = [
                result for result in all_scores
                if isinstance(result.get("difficulty"), str)
                and result["difficulty"].casefold() == difficulty
            ]
            summary_by_difficulty[difficulty] = average_by_variant(difficulty_scores)

        summary_by_difficulty["overall"] = average_by_variant(all_scores)
        summary = {
            "pipeline": type(pipeline).__name__,
            "summary": summary_by_difficulty,
        }
        summary_path = output_path / "summary.json"
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2, ensure_ascii=False)
            summary_file.write("\n")

    return all_scores