import json
import torch
import os
import numpy as np
import random

from ignite.engine import Events
from ignite.contrib.handlers.param_scheduler import CyclicalScheduler


def set_global_seeds(seed):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def attach_metrics(engine, metrics):
    for name, metric in metrics.items():
        metric.attach(engine, name)


class MetricsLogger:
    
    def __init__(self):        
        self.iteration = 0
        
    def attach(self, engine, **kwargs):
        def reset_step(engine):
            engine.state.step = 0
        def inc_step(enine):
            engine.state.step += 1
        engine.add_event_handler(
            event_name=Events.EPOCH_STARTED,
            handler=reset_step
        )
        engine.add_event_handler(
            event_name=Events.ITERATION_STARTED,
            handler=inc_step
        )

    def __call__(
        self,
        engine,
        loader, 
        logger, 
        mode="epoch", 
        validate_every=1,
        loader_name="dev",
        **kwargs
    ):
        evaluator = engine.evaluator
        step = engine.state.epoch if mode == "epoch" else engine.state.step
        if step % validate_every == 0:
            evaluator.loader_name = loader_name
            evaluator.run(loader)
            data = {"mode": mode, "step": step, "loader_name": loader_name}
            data.update(evaluator.state.metrics)
            data.update(kwargs)
            data.update({
                name: values[-validate_every:]
                    for name, values in engine.state.param_history.items()
            })
            logger.info(json.dumps(data))


class OneCycleScheduler(CyclicalScheduler):
    
    def __init__(self, *args, warmup_ratio, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_ratio = warmup_ratio

    def get_param(self):
        cycle_progress = self.event_index / self.cycle_size
        if cycle_progress < self.warmup_ratio:
            mult = cycle_progress / self.warmup_ratio
        else:
            mult = 1 - cycle_progress
        return self.start_value + (self.end_value - self.start_value) * mult


def checkpoint_loader(engine, base_dir=None, to_load=None):
    base_dir = "" if base_dir is None else base_dir
    if to_load is None:
        return
    for filename, obj in to_load.items():
        path = os.path.join(base_dir, filename)
        obj.load_state_dict(torch.load(path))


def iterate_only(engine, n_iters):
    if n_iters > 0 and engine.state.iteration >= n_iters:
        engine.should_terminate = True
