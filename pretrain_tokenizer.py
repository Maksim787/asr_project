import argparse
from string import ascii_lowercase
from pathlib import Path

import sentencepiece as spm

from hw_asr.base.base_text_encoder import BaseTextEncoder
from common import load_train_index


def main(index_directory: str, target_directory: str, vocab_size: int):
    # Check that index directory exists
    index_directory = Path(index_directory)
    assert index_directory.exists()

    # Create target directory to store tokenizer model and vocabulary
    target_directory = Path(target_directory)
    target_directory.mkdir(exist_ok=True)

    # Load datasets
    datasets = load_train_index(index_directory)
    sentences = []
    for dataset in datasets:
        for observation in dataset:
            sentences.append(BaseTextEncoder.normalize_text(observation['text']))

    # Create texts.txt with all texts
    texts_path = target_directory / 'texts.txt'
    with open(texts_path, 'w') as f:
        print(*sentences, sep='\n', file=f)
        # add ascii lowercase letters, so each letter token will be presented in final tokenization
        print(*([' '.join(ascii_lowercase)] * len(sentences)), sep='\n', file=f)

    # Train tokenizer
    model_prefix = target_directory / f'sentence_piece_vocab_{vocab_size}'
    if not model_prefix.with_suffix('.model').exists():
        spm.SentencePieceTrainer.Train(
            input=texts_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe'  # BPE
        )
    else:
        print('Already trained!')
    print('Success')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")

    args.add_argument(
        '-t',
        '--target_directory',
        default='pretrained_model/tokenizer',
        type=str,
        help='Folder with ',
    )
    args.add_argument(
        '-i',
        '--index_directory',
        default='pretrained_model/index/',
        type=str,
        help='Directory with index calculated from the train dataset',
    )
    args.add_argument(
        "-v",
        "--vocab_size",
        default=100,
        type=int,
        help="Test dataset batch size",
    )

    args = args.parse_args()
    main(args.index_directory, args.target_directory, args.vocab_size)
