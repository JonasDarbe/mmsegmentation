# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmengine.hooks.logger_hook import LoggerHook

import neptune

@HOOKS.register_module()
class NeptuneLoggerHook(LoggerHook):
    """Class to log metrics to NeptuneAI.

    It requires `neptune-client` to be installed.

    Args:
        init_kwargs (dict): a dict contains the initialization keys as below:

            - project (str): Name of a project in a form of
              namespace/project_name. If None, the value of NEPTUNE_PROJECT
              environment variable will be taken.
            - api_token (str): User’s API token. If None, the value of
              NEPTUNE_API_TOKEN environment variable will be taken. Note: It is
              strongly recommended to use NEPTUNE_API_TOKEN environment
              variable rather than placing your API token in plain text in your
              source code.
            - name (str, optional, default is 'Untitled'): Editable name of the
              run. Name is displayed in the run's Details and in Runs table as
              a column.

            Check https://docs.neptune.ai/api-reference/neptune#init for more
            init arguments.
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging
        by_epoch (bool): Whether EpochBasedRunner is used.

    .. _NeptuneAI:
        https://docs.neptune.ai/you-should-know/logging-metadata
    """

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 with_step=True,
                 by_epoch=True):

        super(NeptuneLoggerHook, self).__init__(interval=interval, log_metric_by_epoch=False)
        self.init_kwargs = init_kwargs
        self.with_step = with_step
        self.interval = interval

    def before_run(self, runner):
        
        super().before_run(runner)

        self.neptune_run = neptune.init_run(**self.init_kwargs)
        if self.neptune_run is not None:
            self.neptune_run[os.path.basename(runner.cfg.filename)].upload(runner.cfg.filename)
    
    def log(self, data, postfix):
        for k, v in data.items():
            self.neptune_run[f"{k}_{postfix}"].log(v.item())
    
    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]):

        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        
        if batch_idx % self.interval != 0:
            return
        
        self.log(outputs, 'train')
    
    def after_val_epoch(self, runner, metrics):
        
        self.log(metrics, 'val')
        super().after_val_epoch(runner, metrics)

    def after_run(self, runner):
        self.neptune_run.stop()
