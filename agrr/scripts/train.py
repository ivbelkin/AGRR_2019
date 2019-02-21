import argparse
import itertools

from agrr.stuff import *
from agrr.ignite_utils import *

from ignite.engine import Events
from ignite.contrib.handlers import ProgressBar


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
    
    logger.info("[*] Start loading data")
    
    train_loader_train, train_loader_valid = get_loader(conf, "train")
    dev_loader_train, dev_loader_valid = get_loader(conf, "dev")
    
    logger.info("[x] Data loaded")
    logger.info(f"Train on {len(train_loader_train)} batches")
    logger.info(f"Validate on {len(dev_loader_train)} batches")
    logger.info("[*] Model creation started")

    model = get_model(conf)

    logger.info("[x] Model created")
    logger.info("[*] Preparing other stuff")

    optimizer = get_optimizer(conf, model)
    trainer = get_trainer(conf, model, optimizer)
    evaluator = get_evaluator(conf, model)
    training_logger = create_file_logger(conf, "train_logs")
    metrics = get_metrics(conf)
    trainer_pbar = ProgressBar(desc="Train")
    evaluator_pbar = ProgressBar(desc="Valid")
    checkpointer = get_checkpoint_handler(conf)
    scheduler = get_scheduler_handler(
        conf, optimizer, 
        epoch_size=len(train_loader_train)
    )
    freezer = get_freeze_handler(
        conf, model, optimizer,
        epoch_size=len(train_loader_train)
    )
    n_validation_iters = conf["n_validation_iters"] if "n_validation_iters" in conf else -1
    log_metrics_json = MetricsLogger()

    trainer.evaluator = evaluator
    attach_metrics(evaluator, metrics)
    trainer_pbar.attach(trainer, output_transform=lambda x: {"loss": x})
    evaluator_pbar.attach(evaluator)
    log_metrics_json.attach(trainer)

    def reset_should_terminate(e):
        e.should_terminate = False

    evaluator.add_event_handler(
        event_name=Events.STARTED,
        handler=reset_should_terminate,
    )
    evaluator.add_event_handler(
        event_name=Events.ITERATION_COMPLETED,
        handler=iterate_only,
        n_iters=n_validation_iters
    )

#    trainer.add_event_handler(
#        event_name=Events.ITERATION_COMPLETED,
#        handler=log_metrics_json,
#        loader=train_loader_valid,
#        logger=training_logger,
#        mode="iter",
#        validate_every=conf["validation_interval"],
#        loader_name="train"
#    )

    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED,
        handler=log_metrics_json,
        loader=dev_loader_valid,
        logger=training_logger,
        mode="iter",
        validate_every=conf["validation_interval"],
        loader_name="dev"
    )

    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED,
        handler=checkpointer,
        to_save={
            "model": model,
#            "optimizer": optimizer
        },
        save_interval=conf["validation_interval"]
    )

    trainer.add_event_handler(
        event_name=Events.ITERATION_STARTED,
        handler=freezer
    )
    
    trainer.add_event_handler(
        event_name=Events.ITERATION_STARTED,
        handler=scheduler
    )

    logger.info("[x] Stuff created")
    logger.info("[*] Start training")
    total_epoch = sum([
        conf["train_stages"][name]["num_cycles"] * conf["train_stages"][name]["epochs"] 
            for name in conf["train_stages"]["order"]
    ])
    trainer.run(train_loader_train, total_epoch)
    logger.info("[x] Train completed")


if __name__ == "__main__":
    args = parse_args()
    conf = load_config(args.config)
    main(conf)
