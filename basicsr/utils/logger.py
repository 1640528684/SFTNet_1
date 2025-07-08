# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import datetime
import logging
import time

from .dist_util import get_dist_info, master_only


class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    @master_only
    def __call__(self, log_vars):
        """Log training information.
    
        Args:
             log_vars (dict): Contains all variables to be logged.
        """
        # 安全获取所有可能的值（使用一致的变量名）
        current_iter = log_vars.pop('iter', None)  # 兼容不同命名习惯
        total_iter = log_vars.pop('total_iter', current_iter)  # 双重保险
        learning_rate = log_vars.pop('lr', log_vars.pop('learning_rate', None))  # 兼容lr/learning_rate
    
        # 构建日志消息
        log_items = []
        if total_iter is not None:
            log_items.append(f'iter: {int(total_iter):,}')
        if learning_rate is not None:
            log_items.append(f'lr: {float(learning_rate):.3e}')
    
        # 添加其他指标
        for k, v in log_vars.items():
            if isinstance(v, float):
                log_items.append(f'{k}: {v:.4f}')
            elif isinstance(v, int):
                log_items.append(f'{k}: {v:,}')
            else:
                log_items.append(f'{k}: {v}')
    
        # 输出日志
        msg = ' | '.join(log_items)
        self.logger.info(msg)
    
        # TensorBoard记录（保持原逻辑）
        if self.tb_logger:
            for k, v in log_vars.items():
                if isinstance(v, (float, int)):
                    self.tb_logger.add_scalar(k, v, total_iter or 0)
    def reset(self):
        """Reset the logger."""
        self.logger.handlers = []


@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt):
    """We now only use wandb to sync tensorboard log."""
    import wandb
    logger = logging.getLogger('basicsr')

    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb.init(
        id=wandb_id,
        resume=resume,
        name=opt['name'],
        config=opt,
        project=project,
        sync_tensorboard=True)

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')


def get_root_logger(logger_name='basicsr',
                    log_level=logging.INFO,
                    log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    from basicsr.version import __version__
    msg = r"""
                ____                _       _____  ____
               / __ ) ____ _ _____ (_)_____/ ___/ / __ \
              / __  |/ __ `// ___// // ___/\__ \ / /_/ /
             / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/
            /_____/ \__,_//____//_/ \___//____//_/ |_|
     ______                   __   __                 __      __
    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
   / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
  \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
    """
    msg += ('\nVersion Information: '
            f'\n\tBasicSR: {__version__}'
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return msg
