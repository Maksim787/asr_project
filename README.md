# Installation guide

## Install requirements:

```shell
pip install -r requirements.txt
```

## Download model and dataset index:

```shell
python download_model.py
```

This script will download checkpoint and model's config. This script also will download dataset index.

**Arguments:**
1. `--download_directory` (default: `pretrained_model/`) - model and config save directory. Script will also create `index` directory with index inside it.
1. `--config_url`, `model_url`, `index_url` (default: correct links) - Yandex Disk links to download the corresponding files

If you encounter any problems with Yandex Disk API download it manually using default URL arguments from the script.

## Pretrain tokenizer:

```shell
python pretrain_tokenizer.py
```
This script train tokenizer from index and save the model.

**Arguments*:*
1. `--vocab_size` (default: 100) - vocabulary size for BPE tokenizer
1. `--index_directory` (default: `pretrained_model/index`) - directory with dataset index (only train and validation index will be used)
1. `--target_directory` (default: `pretrained_model/tokenizer`) - path to save vocabulary and model.

# Inference guide

## Create predictions with model:

```shell
python test.py -r pretrained_model/model_checkpoint.pth -o output_final
```

**Arguments:**
1. `--config` (default: `None`, it is taken from the model directory) - path to config
1. `--resume` (default: default checkpoint path if some model was trained before) - path to the checkpoint. For the final model it should be provided like in the command above
1. `--device` (default: `None`, use gpu if possible) - the device number for inference
1. `--test-data-folder` (default: `None`, is taken from `test-clean` and `test-other` datasets) - predictions output directory
1. `--batch-size` (default: `20`) - batch size for inference (recommended to set this high for high performece beam search)
1. `--jobs` (default: `1`) - number of workers in dataloaders
1. `--limit` (default: `None`, no limit) - limit the test dataset
1. `--beam_size_ordinary` (default: `20`) - beam size for ordinary beam search
1. `--beam_size_lm` (default: `5000`) - beam size for beam search with language model (it is done in parallel)

## Evaluate predictions:

```shell
python evaluate.py -p output_final
```
Add optional spell checks using argument `--spell_check true`. In this case specify directory with index by argument `--index_directory`.

Metrics will be displayed in the terminal.

**Arguments:**
1. `--predictions_directory` (default: `output_final/`) - directory with predictions. There could be several files, e. g. one for test-clean and one for test-other
1. `--spell_check` (default: `false`) - do the spell check for the predictions
1. `--index_directory` (default: `pretrained_model/index/`) - directory with dataset index (required only for the spell check)


# Train guide

1. To train your own model use:

```shell
python train.py -c pretrained_model/config.json
```

**Arguments:**:
1. `--config` (default: `None`) - path to the model config. _Hint: Some useful configs are in the `hw_asr/configs` directory_.
1. `--resume` (default: `None`) - path to the checkpoint to be resumed
1. `--from_pretrained` (default: `false`) - set to `true` to start training from the beginning. Will take the pretrained model from `--resume` path
1. `--device` (default: `None`) - device (cuda is recommended)

# Miscellaneous

## Tests

Run unit tests using (test for beam search is also implemented):
```shell
python -m unittest discover hw_asr/tests
```

## Jupyter notebooks for models debug

Jupyter notebook used in the process of debugging could be found in `jupyter/` directory.

**Author:** Maksim Kazadaev