import argparse
import os
import pandas as pd
import numpy as np
import csv

from agrr.stuff import *
from agrr.ignite_utils import set_global_seeds
from orgs.agrr_metrics import read_df

from sklearn.model_selection import KFold


def build_args(parser):
    parser.add_argument("--config", type=str, required=True)


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(conf):
    set_global_seeds(conf["seed"])
    
    logger = create_console_logger()
    
    logger.info("[*] Splitiing on {} folds".format(conf["n_folds"]))
    
    df = read_df(os.path.join(conf["input_dir"], conf["input_file"]))
    cv = KFold(n_splits=conf["n_folds"], shuffle=True, random_state=conf["seed"])
    
    os.makedirs(conf["output_dir"], exist_ok=True)
    for fold, (train, valid) in enumerate(cv.split(df)):
        fold_dir = os.path.join(conf["output_dir"], f"fold-{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        df.iloc[train].to_csv(os.path.join(fold_dir, "train.csv"), sep="\t", quoting = csv.QUOTE_NONE)
        df.iloc[valid].to_csv(os.path.join(fold_dir, "valid.csv"), sep="\t", quoting = csv.QUOTE_NONE)

    logger.info("[x] Folds created")


if __name__ == "__main__":
    args = parse_args()
    conf = load_config(args.config)
    main(conf)
