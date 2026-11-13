import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import evaluate
from testing.dataset import load_dataset
from testing.pipeline_def import Pipeline


METRIC_NAMES = ("bleu", "rougeL", "cer", "wer")

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


def _summarize_results(results):
    def average_scores(items):
        return {
            metric: (
                sum(result[metric] for result in items) / len(items)
                if items
                else None
            )
            for metric in METRIC_NAMES
        }

    def average_by_variant(items):
        averages = {"overall": average_scores(items)}
        variants = sorted({
            result["variant"].casefold()
            for result in items
            if isinstance(result.get("variant"), str)
        })
        for variant in variants:
            variant_scores = [
                result for result in items
                if isinstance(result.get("variant"), str)
                and result["variant"].casefold() == variant
            ]
            averages[variant] = average_scores(variant_scores)
        return {"average": averages}

    summary = {}
    difficulties = sorted({
        result["difficulty"].casefold()
        for result in results
        if isinstance(result.get("difficulty"), str)
    })
    for difficulty in difficulties:
        difficulty_scores = [
            result for result in results
            if isinstance(result.get("difficulty"), str)
            and result["difficulty"].casefold() == difficulty
        ]
        summary[difficulty] = average_by_variant(difficulty_scores)

    summary["overall"] = average_by_variant(results)
    return summary


def _write_summary(output_path, pipeline_name, results):
    summary = {
        "pipeline": pipeline_name,
        "summary": _summarize_results(results),
    }
    summary_path = output_path / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)
        summary_file.write("\n")


def rescore_results(
    postprocess_pipeline: Callable[[str], str],
    input_dir="results/raw",
    output_dir="results/postprocessed",
):
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Results directory not found: {input_path}")
    if input_path == output_path or output_path in input_path.parents:
        raise ValueError(
            "Output directory must not be the input directory or its parent"
        )

    results_by_directory = defaultdict(list)
    processed_count = 0
    for source_path in sorted(input_path.rglob("*.json")):
        if source_path.name.casefold() == "summary.json":
            continue
        if output_path in source_path.parents:
            continue

        with source_path.open(encoding="utf-8") as source_file:
            result = json.load(source_file)
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON object in {source_path}")
        if not isinstance(result.get("prediction"), str):
            raise ValueError(f"Missing string prediction in {source_path}")
        if not isinstance(result.get("ground_truth"), str):
            raise ValueError(f"Missing string ground_truth in {source_path}")

        original_prediction = result.get(
            "raw_prediction", result["prediction"]
        )
        prediction = postprocess_pipeline(result["prediction"])
        if not isinstance(prediction, str):
            raise ValueError(
                f"Post-processing pipeline returned a non-string for {source_path}"
            )

        result["raw_prediction"] = original_prediction
        result["prediction"] = prediction
        result.update(score_case(result["ground_truth"], prediction))

        relative_path = source_path.relative_to(input_path)
        destination = output_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as destination_file:
            json.dump(result, destination_file, indent=2, ensure_ascii=False)
            destination_file.write("\n")

        if len(relative_path.parts) >= 4:
            result_directory = Path(*relative_path.parts[:-3])
            pipeline_name = relative_path.parts[-4]
        else:
            result_directory = Path()
            pipeline_name = input_path.name
        results_by_directory[(result_directory, pipeline_name)].append(result)
        processed_count += 1

        _write_summary(
            output_path / result_directory,
            pipeline_name,
            results_by_directory[(result_directory, pipeline_name)],
        )

    return processed_count


def evaluate_pipeline(cases, pipeline, output_dir=None):
    all_scores = []
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
        _write_summary(output_path, type(pipeline).__name__, all_scores)

    return all_scores
