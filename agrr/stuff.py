import yaml
import logging
import torch
import time
import os
import tempfile

from agrr.data_utils import *
from agrr import ignite_utils
from agrr.models import BertAgrrModel
from agrr.scripts.agrr_metrics import read_df, gapping_metrics

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from ignite import engine, utils, metrics, handlers, contrib

from collections import defaultdict


def load_config(filename):
    with open(filename, "r") as f:
        conf = yaml.load(f)
    return conf


def get_tokenizer(conf):
    tokenizer = BertTokenizer.from_pretrained(
        conf["bert_model"], 
        do_lower_case=conf["do_lower_case"]
    )
    return tokenizer


def get_tensors(conf, mode):
    records = load_csv(conf["data_dir"], conf[mode + "_file"])
    tokenizer = get_tokenizer(conf)
    examples = to_examples(records, tokenizer, conf["max_seq_length"])
    token_ids, masks, tag_ids, missed_ids, label_ids, idxs = to_tensor_data(examples)
    return token_ids, masks, tag_ids, missed_ids, label_ids, idxs


def get_loader(conf, mode):
    token_ids, masks, tag_ids, missed_ids, label_ids, idxs = get_tensors(conf, mode)
    if conf["task_name"] == "classification":
        ds = TensorDataset(token_ids, masks, label_ids)
    elif conf["task_name"] == "tagging":
        ds = TensorDataset(token_ids, tag_ids, missed_ids, masks)
    elif conf["task_name"] == "agrr":
        ds = TensorDataset(token_ids, masks, tag_ids, missed_ids, label_ids, idxs)
    else:
        raise NotImplementedError
    train_sampler = RandomSampler(ds)
    valid_sampler = SequentialSampler(ds)
    train_dl = DataLoader(
        ds,
        sampler=train_sampler, 
        batch_size=conf["train" + "_batch_size"],
        num_workers=conf["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    valid_dl = DataLoader(
        ds,
        sampler=valid_sampler, 
        batch_size=conf["dev" + "_batch_size"],
        num_workers=conf["num_workers"],
        pin_memory=True,
        drop_last=False
    )
    return train_dl, valid_dl


def get_device(conf):
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(conf):
    if conf["task_name"] == "classification":
        model = BertForSequenceClassification.from_pretrained(
            conf["bert_model"],
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
            num_labels=2
        )
    elif conf["task_name"] == "tagging":
        model = BertForTokenClassification.from_pretrained(
            conf["bert_model"],
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
            num_labels=7
        )
    elif conf["task_name"] == "agrr":
        model = BertAgrrModel.from_pretrained(
            conf["bert_model"],
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE
        )
    else:
        raise NotImplementedError
    device = get_device(conf)
    logger = logging.getLogger("console")
    logger.info("Using " + device.upper() + " device")
    logger.info(
        f"Number of parameters: {sum([p.nelement() for p in model.parameters()])}", 
    )
    model.to(device)
    return model


def get_optimizer(conf, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer 
                           if not any(nd in n for nd in no_decay)], 
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer 
                           if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=conf["learning_rate"]
    )
    return optimizer


def get_prepare_batch_fn(conf):
    device = get_device(conf)
    if conf["task_name"] == "classification":
        def prepare_batch(batch):
            return {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "labels": batch[2].to(device)
            }
        return prepare_batch
    elif conf["task_name"] == "tagging":
        def prepare_batch(batch):
            return {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[2].to(device),
                "labels": batch[1].to(device)
            }
        return prepare_batch
    elif conf["task_name"] == "agrr":
        def prepare_batch(batch):
            token_ids, masks, tag_ids, missed_ids, label_ids, _ = batch
            return {
                "input_ids": token_ids.to(device),
                "attention_mask": masks.to(device),
                "labels": label_ids.to(device),
                "tags": tag_ids.to(device),
                "gaps": missed_ids.to(device)
            }
        return prepare_batch
    else:
        raise NotImplementedError


def get_trainer(conf, model, optimizer):
    prepare_batch = get_prepare_batch_fn(conf)
    optimizer.zero_grad()
    if conf["task_name"] in ["classification", "tagging", "agrr"]:

        def iter_fn(engine, batch):
            model.train()
            batch = prepare_batch(batch)
            loss = model(**batch) / conf["accumulation_iters"]
            loss.backward()
            if engine.state.iteration % conf["accumulation_iters"] == 0:
                optimizer.step()
                optimizer.zero_grad()
            engine.state.loss = loss.item() * conf["accumulation_iters"]
            return loss.item()

        return engine.Engine(iter_fn)
    else:
        raise NotImplementedError


def get_evaluator(conf, model, compute_metrics=True):
    prepare_batch = get_prepare_batch_fn(conf)

    if conf["task_name"] in ["classification", "tagging"]:
        def iter_fn(engine, batch):
            model.eval()
            with torch.no_grad():
                batch = prepare_batch(batch)
                labels = batch["labels"]
                batch["labels"] = None
                logits = model(**batch)
                return logits, labels

        return engine.Engine(iter_fn)
    elif conf["task_name"] == "agrr":
        def iter_fn(engine, batch):
            model.eval()
            with torch.no_grad():
                idxs = batch[-1]
                batch = prepare_batch(batch)
                labels = batch["labels"]; batch["labels"] = None
                tags = batch["tags"]; batch["tags"] = None
                gaps = batch["gaps"]; batch["gaps"] = None
                logits = model(**batch)
                return logits, idxs

        evaluator = engine.Engine(iter_fn)

        @evaluator.on(engine.Events.STARTED)
        def prepare(e):
            with open(os.path.join(conf["output_dir"], "pred.csv"), "w") as f:
                f.write("\t".join(["text", "class", "cV", "cR1", "cR2", "V", "R1", "R2"]) + "\n")

        @evaluator.on(engine.Events.ITERATION_COMPLETED)
        def save_batch(e):
            sentence_logits, gap_resolution_logits, full_annotation_logits = \
                e.state.output[0]
            idxs = e.state.output[1]

            sentence_logits = sentence_logits.detach().cpu().numpy()
            gap_resolution_logits = gap_resolution_logits.detach().cpu().numpy()
            full_annotation_logits = full_annotation_logits.detach().cpu().numpy()
            idxs = idxs.detach().cpu().numpy()
            
            records = []
            for sl, gl, fl, i in zip(sentence_logits, gap_resolution_logits, full_annotation_logits, idxs):
                records.append(to_result(sl, gl, fl, i))
                
            with open(os.path.join(conf["output_dir"], "pred.csv"), "a") as f:
                for r in records:
                    f.write("\t".join([
                        r["text"], r["class"], r["cV"], r["cR1"], r["cR2"], r["V"], r["R1"], r["R2"]
                    ]) + "\n")
                    
        def eval_metrics(e):
            corr_sents = read_df(os.path.join(conf["data_dir"], conf[e.loader_name + "_file"]))
            test_sents = read_df(os.path.join(conf["output_dir"], "pred.csv"))
            
            size = min(len(corr_sents), len(test_sents))

            corr_sents = corr_sents[:size]
            test_sents = test_sents[:size]
            
            quality = gapping_metrics(corr_sents, test_sents, False)
            
            if not hasattr(e.state, "metrics"):
                e.state.metrics = {}

            e.state.metrics.update(quality)
            
        if compute_metrics:
            evaluator.add_event_handler(
                event_name=engine.Events.COMPLETED,
                handler=eval_metrics
            )

        return evaluator
        
    else:
        raise NotImplementedError


def create_console_logger():
    logger = logging.getLogger("console")
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    formatter = logging.Formatter("{asctime} - {message}", style="{")
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
        

def create_file_logger(conf, logger_name):
    os.makedirs(conf["output_dir"], exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        os.path.join(conf["output_dir"], logger_name + ".log")
    )
    formatter = logging.Formatter("{message}", style="{")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_metrics(conf):
    if conf["task_name"] == "classification":
        def binarize_logits(output):
            logits, labels = output
            return (logits.argmax(dim=1), labels)

        loss = metrics.Loss(torch.nn.CrossEntropyLoss())
        accuracy = metrics.Accuracy(output_transform=binarize_logits)
        precision = metrics.Precision(average=True, output_transform=binarize_logits)
        recall = metrics.Recall(average=True, output_transform=binarize_logits)
        f1 = precision * recall * 2 / (precision + recall + 1e-7)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    elif conf["task_name"] == "tagging":
        def flatten_logits(output):
            logits, labels = output
            return (logits.view(-1, 7), labels.view(-1))

        loss = metrics.Loss(
            torch.nn.CrossEntropyLoss(),
            output_transform=flatten_logits
        )
        return {
            "loss": loss
        }
    elif conf["task_name"] == "agrr":
        return {}
    else:
        raise NotImplementedError


def get_checkpoint_handler(conf):
    if conf["task_name"] == "classification":
        handler = handlers.ModelCheckpoint(
            dirname=conf["output_dir"],
            filename_prefix=time.asctime(),
            score_function=lambda e: e.evaluator.state.metrics["f1_score"],
            score_name="f1",
            n_saved=3,
            atomic=True,
            require_empty=True,
            create_dir=True,
            save_as_state_dict=True
        )

        def wrapper(engine, to_save, save_interval=1):
            if engine.state.iteration % save_interval == 0:
                return handler(engine, to_save)

        return wrapper

    elif conf["task_name"] == "tagging":
        handler = handlers.ModelCheckpoint(
            dirname=conf["output_dir"],
            filename_prefix=time.asctime(),
            score_function=lambda e: e.evaluator.state.metrics["loss"],
            score_name="loss",
            n_saved=3,
            atomic=True,
            require_empty=True,
            create_dir=True,
            save_as_state_dict=True
        )

        def wrapper(engine, to_save, save_interval=1):
            if engine.state.iteration % save_interval == 0:
                return handler(engine, to_save)

        return wrapper
    elif conf["task_name"] == "agrr":
        handler = handlers.ModelCheckpoint(
            dirname=conf["output_dir"],
            filename_prefix=time.asctime(),
            score_function=lambda e: e.evaluator.state.metrics["classification_quality"],
            score_name="f1_score",
            n_saved=3,
            atomic=True,
            require_empty=True,
            create_dir=True,
            save_as_state_dict=True
        )

        def wrapper(engine, to_save, save_interval=1):
            if engine.state.step % save_interval == 0:
                return handler(engine, to_save)

        return wrapper
    else:
        raise NotImplementedError


def get_scheduler_handler(conf, optimizer, epoch_size):
    if conf["task_name"] in ["classification", "tagging", "agrr"]:
        schedulers = []
        durations = []
        for name in conf["train_stages"]["order"]:
            stage = conf["train_stages"][name]
            duration = stage["epochs"] * epoch_size
            scheduler = ignite_utils.OneCycleScheduler(
                optimizer=optimizer,
                param_name="lr",
                start_value=2e-7,
                end_value=stage["learning_rate"],
                cycle_size=duration,
                warmup_ratio=stage["warmup_ratio"]
            )
            schedulers.append(scheduler)
            durations.append(duration * stage["num_cycles"])
        handler = contrib.handlers.ConcatScheduler(
            schedulers=schedulers,
            durations=durations[:-1],
            save_history=True
        )
        return handler
    else:
        raise NotImplementedError


def get_freeze_handler(conf, model, optimizer, epoch_size):
    step = 1
    stages = []
    for name in conf["train_stages"]["order"]:
        stage = conf["train_stages"][name]
        groups = stage["groups"] if "groups" in stage else None
        stages.append((step, groups))
        step += stage["num_cycles"] * stage["epochs"] * epoch_size

    def reset_bert_adam():
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                optimizer.state[p] = defaultdict(dict)

    def handler(engine):
        groups = []
        for stage in stages:
            if stage[0] == engine.state.iteration:
                groups = stage[1]
        if groups is None:
            print("Iteration:", engine.state.iteration)
            print("Training all groups")
            reset_bert_adam()
            for p in model.parameters():
                p.requires_grad = True
        elif len(groups) == 0:
            return
        else:
            print("Iteration:", engine.state.iteration)
            print("Training:", groups, "groups")
            reset_bert_adam()
            for p in model.parameters():
                p.requires_grad = False
            for group in groups:
                for p in getattr(model, group).parameters():
                    p.requires_grad = True

    return handler
