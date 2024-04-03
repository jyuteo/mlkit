import os
import wandb

from typing import List, Dict, Optional, Union

from .logger import Logger

WANDB_ENABLED = False


class WandBLogger:
    REQUIRED_ENV_VARS = ["WANDB_API_KEY"]

    def __init__(
        self,
        logger: Logger,
        project: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[List] = None,
        name: Optional[str] = None,
        job_type: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        entity: Optional[str] = None,
        save_code: Optional[bool] = None,
        is_master_process: bool = True,
        **kwargs,
    ):
        self.is_master_process = is_master_process
        self.logger = logger
        self.project = project
        self.group = group
        self.tags = tags
        self.name = name
        self.job_type = job_type
        self.config = config
        self.entity = entity
        self.save_code = save_code

    def _disable_wandb(self, msg: str = None):
        if not self.is_master_process:
            return
        global WANDB_ENABLED
        WANDB_ENABLED = False
        self.logger.info(f'WANDB is disabled {f": {msg}" if msg is not None else ""}')

    def _is_enabled(self):
        if not self.is_master_process:
            return
        return WANDB_ENABLED

    def login(self) -> bool:
        if not self.is_master_process:
            return
        if any(
            wandb_env_var not in os.environ for wandb_env_var in self.REQUIRED_ENV_VARS
        ):
            WandBLogger._disable_wandb("Unable to get required env variables for WandB")
            return False
        success = wandb.login(
            key=os.environ["WANDB_API_KEY"],
        )
        if not success:
            WandBLogger._disable_wandb("Login failed")
            return False
        self.logger.info("WandB login successful")
        return True

    def start(self):
        if not self.is_master_process:
            return
        wandb.init(
            project=self.project,
            group=self.group,
            tags=self.tags,
            name=self.name,
            job_type=self.job_type,
            config=self.config,
            entity=self.entity,
            save_code=self.save_code,
            force=True,
        )
        global WANDB_ENABLED
        WANDB_ENABLED = True
        self.logger.info(
            f"WANDB run starting with id {wandb.run.id} and name {wandb.run.name}."
        )

    def log(self, *args, **kwargs):
        if not self.is_master_process:
            return
        if self._is_enabled():
            return wandb.log(*args, **kwargs)

    def log_metrics(self, metrics: Dict, step: int, category: str = ""):
        if not self.is_master_process:
            return
        if not category:
            self.log(metrics, step=step)
        else:
            assert isinstance(category, str), "Category must be of type string"
            self.log({category: metrics}, step=step)
