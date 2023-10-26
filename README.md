# ASR project barebones

## Installation guide

1. Install requirements:

```shell
pip install -r requirements.txt
```

2. Download model and dataset index:

```shell
python download_model.py
```

This script will download checkpoint and model's config in `pretrained_model/` directory (can by modified by `--download_directory` argument). This script also will download dataset index in `pretrained_model/index/` directory.

3. Pretrain tokenizes:

```shell
python pretrain_tokenizer.py
```

This script pretrain tokenizer from index in `pretrained_model/index/` and put the model inside `pretrained_model/tokenizer` (both directories could be modified by corresponding arguments).

4. Generate predictions with model:


```shell
python test.py -r pretrained_model/model_checkpoint.pth -o output_final
```

5. Evaluate predictions:

```shell
python evaluate.py -p output_final
```
Add optional spell checks using argument `--spell_check true`. In this case specify directory with index by argument `--index_directory`.

Metrics will be displayed in the terminal.

6. To train you own model use:

```shell
python train.py -c pretrained_model/config.json
```
