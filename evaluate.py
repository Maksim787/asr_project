import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm
from spellchecker import SpellChecker
from textblob import TextBlob

from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer, calc_wer


def get_words_from_index(ind: list[dict]):
    cnt = Counter()
    for observation in ind:
        cnt.update(BaseTextEncoder.normalize_text(observation['text']).split())
    return cnt


def build_dictionary(datasets: list[dict]):
    word_counts = get_words_from_index(sum(datasets, start=[]))

    spell = SpellChecker()
    spell.word_frequency.remove_words(spell.word_frequency.dictionary)
    words = []
    for word, count in word_counts.items():
        words += [word] * count
    spell.word_frequency.load_words(words)
    assert spell.word_frequency.unique_words == len(word_counts)
    return spell


def correct_by_dataset_words(text: str, spell: SpellChecker):
    result = []
    for word in text.split():
        corrected = spell.correction(word)
        if corrected:
            result.append(corrected)
        else:
            result.append(word)
    return ' '.join(result)


def correct_by_common_words(sentence):
    return str(TextBlob(sentence).correct())


def add_predictions(observation: dict[str, str], spell):
    observation = observation.copy()
    best_prediction = next(filter(lambda x: 'beam_search_lm' in x, observation.keys()))
    observation['pred_text_corrected_by_datasets'] = correct_by_dataset_words(observation[best_prediction], spell)
    observation['pred_text_corrected_by_common_words'] = correct_by_common_words(observation[best_prediction])
    return observation


def compute_metrics(predictions: dict[str, list[dict[str, str]]], spell: SpellChecker) -> dict[str, dict[str, dict[str, float]]]:
    metrics_by_part = {}
    for part, observations in predictions.items():
        metrics = {'WER': defaultdict(list), 'CER': defaultdict(list)}
        for observation in tqdm(observations):
            observation = add_predictions(observation, spell)
            ground_truth = observation['ground_truth']
            for name, prediction in observation.items():
                if name != 'ground_truth':
                    for metric, func in zip(['WER', 'CER'], [calc_wer, calc_cer]):
                        metrics[metric][name].append(func(ground_truth, prediction) * 100)
                        metrics[metric][name].append(func(ground_truth, prediction) * 100)
        metrics_by_part[part] = {
            metric: {
                name: np.array(values).mean() for name, values in metric_by_predictions.items()
            } for metric, metric_by_predictions in metrics.items()
        }
    return metrics_by_part


def print_metrics(metrics: dict[str, dict[str, dict[str, float]]]):
    for part, metrics_by_part in metrics.items():
        print('=' * 10)
        print(part)
        df = pd.DataFrame(columns=metrics_by_part.keys())
        for metric_name, metric_by_predictions in metrics_by_part.items():
            for predictions_name, metric_value in metric_by_predictions.items():
                df.loc[predictions_name, metric_name] = metric_value
        print(df)
        print('=' * 10)


def main(args):
    index_directory = Path(args.index_directory)
    assert index_directory.exists()
    predictions_directory = Path(args.predictions_directory)
    assert predictions_directory.exists()

    # Load observations from train datasets
    datasets = []
    for path in index_directory.iterdir():
        if ('train' in path.name or 'dev' in path.name) and path.name.endswith('.json'):
            with open(path, 'r') as f:
                datasets.append(json.load(f))
    assert len(datasets) == 5

    # Construct dictionary
    spell = build_dictionary(datasets)

    # Load predictions
    predictions = {}
    for path in predictions_directory.iterdir():
        with open(path, 'r') as f:
            predictions[f'{path.name.removesuffix("_output.json")}'] = json.load(f)

    metrics = compute_metrics(predictions, spell)
    print_metrics(metrics)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    
    args.add_argument(
        "-i",
        "--index_directory",
        default='saved_server/index/',
        type=str,
        help="train + validation datasets path",
    )

    args.add_argument(
        "-p",
        "--predictions_directory",
        default='output_final/',
        type=str,
        help="train + validation datasets path",
    )

    main(args.parse_args())
