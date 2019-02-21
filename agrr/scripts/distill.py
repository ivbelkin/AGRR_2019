import argparse
import os
import pandas as pd
import numpy as np
import csv
import tqdm

from agrr.stuff import *
from agrr.ignite_utils import set_global_seeds
from orgs.agrr_metrics import read_df

from sklearn.model_selection import KFold

np.warnings.filterwarnings("ignore")


def build_args(parser):
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--preds_dir", type=str, required=True)


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(conf):    
    logger = create_console_logger()
    
    logger.info(f"[*] Disstilation for {conf.input_file} started")
    
    preds = []
    orig_df = read_df(os.path.join(conf.data_dir, conf.input_file))
    for file in os.listdir(conf.preds_dir):
        if not file.startswith(conf.input_file):
            continue
        path = os.path.join(conf.preds_dir, file)
        logger.info(f"[*] Loading {path}")
        preds.append(read_df(path))
    logger.info(f"[x] Loaded {len(preds)} files")
    
    for df in preds:
        assert len(df) == len(orig_df)
    
    rows = []
    for i in tqdm.tqdm(range(len(df))):
        added = False
        for df in preds:
            if (df.iloc[i, 1:] == orig_df.iloc[i, 1:]).all():
                rows.append(orig_df.iloc[i])
                added = True
            if added:
                break
        if added:
            continue
        for df1 in preds:
            if df1.iloc[i]["class"] == "":
                continue
            cnt = 0
            for df2 in preds:
                if df1.iloc[i]["class"] == "":
                    continue
                if (df1.iloc[i, 1:] == df2.iloc[i, 1:]).all():
                    cnt += 1
            if 2 * cnt > len(preds):
                row = df1.iloc[i]
                row.loc["text"] = orig_df.iloc[i]["text"]
                rows.append(row)
                added = True
            if added:
                break
    
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(conf.preds_dir, conf.input_file),
        sep="\t", quoting = csv.QUOTE_NONE, index=False
    )
    logger.info(f"[x] Distilled {len(df) / len(orig_df)}% from {conf.input_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
