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
1. `config_url`, `model_url`, `index_url` (default: correct links) - Yandex Disk links to download the corresponding files

If you encounter any problems with Yandex Disk API download it manually using default URL arguments from the script.

## Pretrain tokenizer:

```shell
python pretrain_tokenizer.py
```
This script train tokenizer from index and save the model.

**Arguments*:*
1. `vocab_size` (default: 100) - vocabulary size for BPE tokenizer
1. `index_directory` (default: `pretrained_model/index`) - directory with dataset index (only train and validation index will be used)
1. `target_directory` (default: `pretrained_model/tokenizer`) - path to save vocabulary and model.

# Inference guide

## Create predictions with model:

```shell
python test.py -r pretrained_model/model_checkpoint.pth -o output_final
```

TODO: arguments

## Evaluate predictions:

```shell
python evaluate.py -p output_final
```
Add optional spell checks using argument `--spell_check true`. In this case specify directory with index by argument `--index_directory`.

Metrics will be displayed in the terminal.

TODO: arguments


# Train guide

1. To train your own model use:

```shell
python train.py -c pretrained_model/config.json
```

TODO: arguments

**Author:** Maksim Kazadaev