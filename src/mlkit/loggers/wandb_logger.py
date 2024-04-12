import os
import wandb
import pandas as pd

from typing import List, Dict, Optional, Union

from .logger import Logger


class WandBLogger:
    REQUIRED_ENV_VARS = ["WANDB_HOST", "WANDB_API_KEY"]
    WANDB_TABLE_MAX_ROW = 2000

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
        self._is_enabled = False

    def _disable_wandb(self, msg: str = None):
        if not self.is_master_process:
            return
        self._is_enabled = False
        self.logger.info(f'WANDB is disabled {f": {msg}" if msg is not None else ""}')

    def login(self) -> bool:
        if not self.is_master_process:
            return
        if any(
            wandb_env_var not in os.environ for wandb_env_var in self.REQUIRED_ENV_VARS
        ):
            self._disable_wandb("Unable to get required env variables for WandB")
            return False
        success = wandb.login(
            key=os.environ["WANDB_API_KEY"],
            host=os.environ["WANDB_HOST"],
            relogin=True,
            force=True,
        )
        if not success:
            self._disable_wandb("Login failed")
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
        self._is_enabled = True
        self.logger.info(
            f"WANDB run starting with id {wandb.run.id} and name {wandb.run.name}."
        )

    def close(self):
        if not self.is_master_process:
            return
        wandb.finish()
        self.logger.info("Wandb closed.")

    def log(self, *args, **kwargs):
        if not self.is_master_process:
            return
        if self._is_enabled:
            return wandb.log(*args, **kwargs)

    def log_metrics(self, metrics: Dict, step: int, category: str = ""):
        if not self.is_master_process:
            return
        if not category:
            self.log(metrics, step=step)
        else:
            assert isinstance(category, str), "Category must be of type string"
            metrics_with_category = {}
            for key, val in metrics.items():
                metrics_with_category[f"{category}/{key}"] = val
            self.log(metrics_with_category, step=step)

    def log_table(self, table: Union[wandb.Table, pd.DataFrame], table_name: str):
        if isinstance(table, pd.DataFrame):
            table = wandb.Table(dataframe=table)
        elif not isinstance(table, wandb.Table):
            raise f"Invalid table type, expect wandb.Table or pd.DataFrame, but got {type(table)}"
        table_chunks = table.data // self.WANDB_TABLE_MAX_ROW
        for i in range(table_chunks + 1):
            start, end = (
                i * self.WANDB_TABLE_MAX_ROW,
                i * self.WANDB_TABLE_MAX_ROW + self.WANDB_TABLE_MAX_ROW,
            )
            chunk = table.data[start:end]
            table_name = table_name if i == 0 else f"{table_name} - {i + 1}"
            wandb.log({table_name: wandb.Table(data=chunk)})
