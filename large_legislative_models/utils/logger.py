""" 
This Python script defines a custom logger class Logger that extends from LoguruLogger. It includes a decorator requires_cfg to ensure the logger is configured before use.

The Logger class has methods to configure the logger (cfg), log data later (log_later), flush the logged data (flush), and close the logger (close).

The logger can log locally, to a file, or to Weights & Biases (wandb), depending on the configuration.

"""

from typing import Any, Dict, Optional
from datetime import datetime

import os
import sys
from collections import ChainMap
from io import StringIO

import torch
from loguru._logger import Core
from loguru._logger import Logger as LoguruLogger
from rich.console import Console
from rich.table import Table

from large_legislative_models import Config
from large_legislative_models.globals import Globals


def requires_cfg(func):
    def wrapper(self, *args, **kwargs):
        if not self.configured:
            raise ValueError("Logger must be configured with `logger.cfg` before using this function")
        return func(self, *args, **kwargs)

    return wrapper


class Logger(LoguruLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configured = False

    def cfg(self, cfg: Config):
        self.log_wandb = cfg.log_wandb
        self.buffer = []
        self._flush_step = 0
        self.wandb_project = cfg.wandb_project_name
        self.log_locally = cfg.log_locally

        current_time = datetime.now()
        # Convert to string with your desired format
        time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S.%f")

        self.filename = f"{Globals.LOG_DIR}/{time_string}random_results.txt"

        if self.log_wandb:
            import wandb

            self.wandb_module = wandb  # a hack to lazy load modules. this is unfortunate but necessary
            if self.wandb_project is None:
                raise ValueError("wandb_project must be provided when log_wandb is True")
            if cfg.wandb_tags != "":
                wandb.init(project=cfg.wandb_project_name, entity=cfg.wandb_entity, config=cfg, tags=[cfg.wandb_tags])
            else:
                wandb.init(project=cfg.wandb_project_name, entity=cfg.wandb_entity, config=cfg)

            wandb.define_metric("principal_final/principal_step")
            wandb.define_metric("principal_final/*", step_metric=f"principal_final/principal_step")

            wandb.define_metric("train/episode")
            wandb.define_metric("train/*", step_metric="train/episode")

            wandb.define_metric("validation/episode")
            wandb.define_metric("validation/*", step_metric="validation/episode")

            wandb.define_metric("combined_val_train/episode")

            wandb.define_metric("opt/step")
            wandb.define_metric("opt/*", step_metric="opt/step")

        if not cfg.log_locally:
            self.remove()

        if cfg.log_file:
            if isinstance(cfg.log_file, bool):
                cfg.log_file = "output.log"

            log_directory = os.path.dirname(cfg.log_file)

            if log_directory != "" and not os.path.exists(log_directory):
                os.makedirs(log_directory)
            self.log_directory = log_directory
            self.add(cfg.log_file, rotation="10 MB")  # what happens on rotation?
            print("Logging to:", os.path.abspath(cfg.log_file))

        self.configured = True

    @requires_cfg
    def log_later(
        self,
        data: Optional[Dict[str, Any]] = None,
        flush: bool = False,
    ):
        # Add to buffer
        self.buffer.append(data)
        if flush:
            self.flush(call_stack_depth=4)

    @requires_cfg
    def flush(self, call_stack_depth=2):
        data = dict(ChainMap(*self.buffer))  # convert list of dicts to single dict, overwriting values with later dicts

        # Log to console by converting to table
        table = Table("Key", "Value", title=f"Log Step {self._flush_step}")

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().detach().tolist()

            if isinstance(value, float):
                value = round(value, 4)
            table.add_row(str(key), str(value))

        out = Console(file=StringIO())
        if self.log_locally:
            out.print(table)

        # need to log four levels above: flush, requires_cfg, log_later, requires_cfg
        self.opt(depth=call_stack_depth).info(f"\n{out.file.getvalue()}")

        if self.log_wandb:
            # Log to wandb
            self.wandb_module.log(data)

        # Clear the buffer
        self.buffer.clear()
        self._flush_step += 1

    @requires_cfg
    def log_video(self, video_path, episode):
        if self.log_wandb:
            self.wandb_module.log({f"video": self.wandb_module.Video(data_or_path=video_path)})

    def log_distribution(self, dist, head_id, principal_step, epoch, x_axis="probability"):
        if self.log_wandb:
            data = [[i, p] for i,p in enumerate(dist)]
            table = self.wandb_module.Table(data=data, columns=["action", f"{x_axis}"])
            self.wandb_module.log(
                {f"{principal_step+epoch}; Principal step {principal_step} - epoch {epoch}: bracket {head_id}": 
                self.wandb_module.plot.bar(
                    table,
                    "action",
                    f"{x_axis}",
                    title=f"{principal_step+epoch}; Principal step {principal_step} - epoch {epoch}: bracket {head_id}")})

    @requires_cfg
    def log_prompt_and_response(self, prompt, response, generation_number):
        # TODO: clean (wandb truncates lines over 1000 characters)
        def chunk_string(s, chunk_size=900):
            prompt_chunks = [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]
            return "\n".join(prompt_chunks)

        log_entry = (
            f"True Generation {generation_number}: \n Prompt: {chunk_string(prompt)}, \n Response: {response}\n \n"
        )
        with open(self.filename, "a") as f:
            f.write(log_entry)

        if self.log_wandb:
            self.wandb_module.save(self.filename, base_path=os.path.dirname(self.filename), policy="now")

    @requires_cfg
    def log_random_results(self, taxes, p_return, std, p_step):
        # TODO: clean (wandb truncates lines over 1000 characters)
        log_entry = f"{taxes} -- {p_return} -- {std} -- {p_step}\n"

        with open(self.filename, "a") as f:
            f.write(log_entry)

        if self.log_wandb:
            self.wandb_module.save(self.filename, base_path=os.path.dirname(self.filename), policy="now")


logger = Logger(
    core=Core(),
    exception=None,
    depth=0,
    record=False,
    lazy=False,
    colors=False,
    raw=False,
    capture=True,
    patchers=[],
    extra={},
)

if sys.stderr:
    logger.add(sys.stderr)


if __name__ == "__main__":
    """
    Sample usage: logger is used to log a message, configure itself with a Config instance, and log another message with immediate flushing.
    """
    logger.info({"hi": "Logger cfg"})
    logger.cfg(Config())
    logger.log_later({"hi": "Logger cfg"}, flush=True)
