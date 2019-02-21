import argparse
import os
import csv

from agrr.stuff import *
from agrr.ignite_utils import *

from ignite.engine import Events
from ignite.contrib.handlers import ProgressBar

from orgs.agrr_metrics import read_df


def build_args(parser):
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(conf):
    logger = create_console_logger()
    
    logger.info("[*] Start loading data")
    
    _, dev_loader_valid = get_loader(conf, "dev")
    
    logger.info("[x] Data loaded")
    logger.info(f"Inference on {len(dev_loader_valid)} batches")
    logger.info("[*] Model creation started")

    model = get_model(conf)

    logger.info("[x] Model created")
    checkpoints = []
    if conf["checkpoint"] == "all":
        for file in os.listdir(conf["output_dir"]):
            if "model" in file:
                checkpoints.append(file)
    else:
        checkpoints.append(conf["checkpoint"])

    logger.info("[*] Start inference")
    for checkpoint in checkpoints:
        logger.info("[*] Using checkpoint '" + checkpoint + "'")
        evaluator = get_evaluator(conf, model, compute_metrics=False)
        evaluator_pbar = ProgressBar(desc="Valid")
        evaluator_pbar.attach(evaluator)

        evaluator.add_event_handler(
            event_name=Events.STARTED,
            handler=checkpoint_loader,
            base_dir=conf["output_dir"],
            to_load={
                checkpoint: model
            }
        )
        evaluator.run(dev_loader_valid)
        orig_df = read_df(os.path.join(conf["data_dir"], conf["dev_file"]))
        pred_df = read_df(os.path.join(conf["output_dir"], "pred.csv"))
        pred_df["text"] = orig_df["text"]
        pred_df.to_csv(
            os.path.join(conf["output_dir"], conf["dev_file"] + "_" + checkpoint),
            index=False, sep="\t", quoting=csv.QUOTE_NONE
        )
    logger.info("[x] Inference completed")


if __name__ == "__main__":
    args = parse_args()
    conf = load_config(args.config)
    conf["dev_file"] = args.input_file
    conf["checkpoint"] = args.checkpoint
    main(conf)
