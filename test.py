import argparse
import json
import os
import multiprocessing
from pathlib import Path


import torch
import numpy as np
from tqdm import tqdm

import hw_asr.model as module_model
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_dir):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    text_encoder = config.get_text_encoder()
    # load LM
    text_encoder.load_lm()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    BEAM_SIZE_FOR_BEAM_SEARCH = 20
    BEAM_SIZE_FOR_LM_BEAM_SEARCH = 2000

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        with torch.no_grad():
            for part, dataloader in dataloaders.items():
                out_file = out_dir / f"{part}_output.json"
                results = []
                for batch_num, batch in enumerate(tqdm(dataloader)):
                    batch = Trainer.move_batch_to_device(batch, device)
                    output = model(**batch)
                    if type(output) is dict:
                        batch.update(output)
                    else:
                        batch["logits"] = output
                    batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
                    batch["log_probs_length"] = model.transform_input_lengths(batch["spectrogram_length"])
                    batch["probs"] = batch["log_probs"].exp().cpu()
                    batch["argmax"] = batch["probs"].argmax(-1)
                    pred_text_beam_search_lm = text_encoder.ctc_beam_search_lm(batch["log_probs"], batch["log_probs_length"], pool=pool, beam_size=BEAM_SIZE_FOR_LM_BEAM_SEARCH)
                    for i in range(len(batch["text"])):
                        argmax = batch["argmax"][i]
                        argmax = argmax[: int(batch["log_probs_length"][i])]
                        pred_text_beam_search = text_encoder.ctc_beam_search(batch["log_probs"][i], batch["log_probs_length"][i], beam_size=BEAM_SIZE_FOR_BEAM_SEARCH)[0].text
                        results.append(
                            {
                                "ground_truth": batch["text"][i],
                                "pred_text_argmax": text_encoder.ctc_decode_enhanced(argmax.cpu().numpy()),
                                f"pred_text_beam_search_{BEAM_SIZE_FOR_BEAM_SEARCH}": pred_text_beam_search,
                                f"pred_text_beam_search_lm_{BEAM_SIZE_FOR_LM_BEAM_SEARCH}": pred_text_beam_search_lm[i]
                            }
                        )

                with Path(out_file).open("w") as f:
                    json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output_dir",
        default="output",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )
    args.add_argument(
        "-l",
        "--limit",
        default=None,
        type=int,
        help="Limit the test set"
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                        "module": "hw_asr.datasets"
                    }
                ],
            }
        }
    else:
        config.config["data"] = {
            "test-clean": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "LibrispeechDataset",
                        "args": {
                            "part": "test-clean",
                            "limit": args.limit
                        },
                        "module": "hw_asr.datasets"
                    }
                ],
            },
            "test-other": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "LibrispeechDataset",
                        "args": {
                            "part": "test-other",
                            "limit": args.limit
                        },
                        "module": "hw_asr.datasets"
                    }
                ],
            }
        }

    main(config, args.output_dir)
