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
from common import load_train_index


def get_words_from_index(ind: list[dict]) -> Counter:
    """
    Count words in the dataset
    """
    cnt = Counter()
    for observation in ind:
        # Normalize text before splitting for words
        cnt.update(BaseTextEncoder.normalize_text(observation['text']).split())
    return cnt


def build_dictionary(datasets: list[dict]) -> SpellChecker:
    """
    Build dictionary from datasets index
    """
    # Count words
    word_counts = get_words_from_index(sum(datasets, start=[]))

    # Construct spell checker without pretrained words
    spell = SpellChecker()
    spell.word_frequency.remove_words(spell.word_frequency.dictionary)
    # Add words from datasets
    words = []
    for word, count in word_counts.items():
        words += [word] * count
    spell.word_frequency.load_words(words)
    assert spell.word_frequency.unique_words == len(word_counts)
    return spell


def correct_by_dataset_words(text: str, spell: SpellChecker):
    """
    Correct text using dictionary from dataset
    """
    result = []
    # Iterate over words
    for word in text.split():
        corrected = spell.correction(word)
        if corrected:
            result.append(corrected)  # word was found in the dictionary
        else:
            result.append(word)  # no close words to the given word
    return ' '.join(result)


def correct_by_common_words(sentence: str) -> str:
    """
    Correct by the dictionary from the internet
    """
    # Correct sentence and normalize it
    return BaseTextEncoder.normalize_text(str(TextBlob(sentence).correct()))


def add_predictions(observation: dict[str, str], spell: SpellChecker | None, best_field: str = 'beam_search_lm'):
    """
    Add predictions with spell checks
    """
    if spell is None:
        return observation
    observation = observation.copy()
    # Find field to correct (by default it is the beam search with language model)
    best_prediction = next(filter(lambda x: best_field in x, observation.keys()))
    # Correct predictions by dictionary from datasets
    observation['pred_text_corrected_by_dataset'] = correct_by_dataset_words(observation[best_prediction], spell)
    # Correct predictions by dictionary from the internet
    observation['pred_text_corrected_by_common_words'] = correct_by_common_words(observation[best_prediction])
    return observation


def compute_metrics(predictions: dict[str, list[dict[str, str]]], spell: SpellChecker) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute WER and CER for each part of predictions
    """
    metrics_by_part = {}
    # Iterate over parts of predictions (e. g. test-clean and test-other)
    for part, list_of_observations_for_part in predictions.items():
        # Construct list of metrics (metric values for each sample)
        metrics = {'WER': defaultdict(list), 'CER': defaultdict(list)}
        # Iterate over samples (sample is the dictionary of ground_truth and different type predictions)
        for sample in tqdm(list_of_observations_for_part):
            # Add corrected predictions for spell check if necessary
            sample = add_predictions(sample, spell)
            ground_truth = sample['ground_truth']
            # Iterate over predictions types
            for prediction_type, prediction in sample.items():
                if prediction_type == 'ground_truth':
                    continue
                # Calculate WER and CER
                for metric_name, func in zip(['WER', 'CER'], [calc_wer, calc_cer]):
                    metrics[metric_name][prediction_type].append(func(ground_truth, prediction) * 100)
        # Average metrics for each prediction type across all samples
        metrics_by_part[part] = {
            metric_name: {
                name: np.array(values).mean() for name, values in metrics_by_prediction_type.items()
            } for metric_name, metrics_by_prediction_type in metrics.items()
        }
    return metrics_by_part


def print_metrics(metrics: dict[str, dict[str, dict[str, float]]]):
    """
    Construct pd.DataFrame and print it in human-readable format
    """
    # Iterate over parts (e. g. test-clean and test-other)
    for part, metrics_by_part in metrics.items():
        print('=' * 10)
        print(part)
        # Construct pd.DataFrame with metric names (WER and CER)
        df = pd.DataFrame(columns=metrics_by_part.keys())
        # Iterate over metrics (WER and CER)
        for metric_name, metric_by_prediction_type in metrics_by_part.items():
            # Iterate over prediction types (argmax, beam search, etc.)
            for predictions_name, metric_value in metric_by_prediction_type.items():
                df.loc[predictions_name, metric_name] = metric_value
        # Print results
        print(df.to_string())
        print('=' * 10)


def main(predictions_directory: str, spell_check: bool, index_directory: str):
    # Check predictions directory existence
    predictions_directory = Path(args.predictions_directory)
    assert predictions_directory.exists()

    # Construct spell dictionary from dataset if necessary
    if spell_check:
        # Check that index directory exists
        index_directory = Path(index_directory)
        assert index_directory.exists()

        # Load index
        datasets = load_train_index(index_directory)

        # Construct dictionary
        spell = build_dictionary(datasets)
    else:
        spell = None

    # Load predictions
    predictions = {}
    for path in predictions_directory.iterdir():
        with open(path, 'r') as f:
            predictions[f'{path.name.removesuffix("_output.json")}'] = json.load(f)

    # Compute and print metrics
    metrics = compute_metrics(predictions, spell)
    print_metrics(metrics)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument(
        '-p',
        '--predictions_directory',
        default='output_final/',
        type=str,
        help='Path to directory with files {name}_output.json',
    )

    args.add_argument(
        '-s',
        '--spell_check',
        default='false',
        type=str,
        help='Do a spell check'
    )

    args.add_argument(
        '-i',
        '--index_directory',
        default='pretrained_model/index/',
        type=str,
        help='Directory with index calculated from the train dataset',
    )

    args = args.parse_args()
    main(args.predictions_directory, args.spell_check.lower() == 'true', args.index_directory)
