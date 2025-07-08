import argparse
import datetime
import logging
import math
import os
import random
import sys
import time

from os import path as osp
root_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录（basicsr）
sys.path.append(os.path.dirname(root_dir))  # 将根目录（NAFNet）加入路径
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str, init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse
import math
from basicsr.data.transforms import (  # 直接从transforms.py导入
    PairedRandomCrop,
    Augment,
    AdjustSize,
    Compose,
    Resize
)
import basicsr.data.transforms as transforms

#重新初始化 cuDNN
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # 分布式设置
    if args.launcher == 'none':
        opt['dist'] = False
    else:
        opt['dist'] = True
        init_dist(args.launcher, **opt['dist_params'] if 'dist_params' in opt else {})
    opt['rank'], opt['world_size'] = get_dist_info()

    # 随机种子
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    # 设备设置
    opt['num_gpu'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    opt['device'] = 'cuda' if opt['num_gpu'] > 0 else 'cpu'
    
    # 优化器调整
    if is_train and opt['train'].get('optim_g', {}).get('type') == 'AdamW':
        opt['train']['optim_g'].setdefault('weight_decay', 0.0001)
        opt['train']['optim_g'].setdefault('betas', (0.9, 0.999))

    return opt

def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    
    # 记录硬件信息
    logger.info(get_env_info())
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    if torch.cuda.is_available():
        logger.info(f"GPU Name: {torch.cuda.get_device_name(opt.get('local_rank', 0))}")
    
    logger.info(dict2str(opt))

    # 初始化 WandB 和 TensorBoard
    if opt['logger'].get('wandb') and opt['logger']['wandb'].get('project'):
        init_wandb_logger(opt)
    tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name'])) if opt['logger'].get('use_tb_logger') else None
    return logger, tb_logger

def adjust_image_size(img, patch_size=16):
    """调整图像尺寸为 patch_size 的整数倍"""
    ow, oh = img.size
    h = int(math.ceil(oh / patch_size) * patch_size)
    w = int(math.ceil(ow / patch_size) * patch_size)
    if (h == oh) and (w == ow):
        return img
    return transforms.Resize((h, w))(img)

def create_train_val_dataloader(opt, logger):
    train_loader = val_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        transform_list = [
            AdjustSize(patch_size=dataset_opt.get('patch_size', 16))
        ]
        if 'transforms' in dataset_opt:
            transform_list.extend([getattr(transforms, t[0])(**t[1]) for t in dataset_opt['transforms']])
        dataset_opt['transform'] = Compose(transform_list)

        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'],
                sampler=train_sampler, seed=opt['manual_seed'])
            
            # 梯度累积适配
            batch_size = dataset_opt['batch_size_per_gpu'] * opt['world_size']
            num_iter_per_epoch = math.ceil(len(train_set) * dataset_enlarge_ratio / batch_size)
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            
            logger.info(
                f"Training data statistics:\n"
                f"\tNumber of images: {len(train_set)}\n"
                f"\tBatch size (total): {batch_size}\n"
                f"\tWorld size: {opt['world_size']}\n"
                f"\tTotal epochs: {total_epochs}, iters: {total_iters}")

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'],
                sampler=None, seed=opt['manual_seed'])
            logger.info(f'Validation images: {len(val_set)}')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters

def main():
    opt = parse_options(is_train=True)
    torch.backends.cudnn.benchmark = True

    # 自动恢复训练
    if opt['rank'] == 0:
        state_dir = osp.join('experiments', opt['name'], 'training_states')
        if osp.exists(state_dir):
            states = [f for f in os.listdir(state_dir) if f.endswith('.state')]
            if states:
                latest_state = max(states, key=lambda x: int(x.split('.')[0]))
                opt['path']['resume_state'] = osp.join(state_dir, latest_state)

    # 初始化模型
    model = create_model(opt)
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
        model.resume_training(resume_state)
        start_epoch = resume_state.get('epoch', 0)
        current_iter = resume_state.get('iter', 0)
        logger.info(f"Resumed from epoch {start_epoch}, iter {current_iter}")
    else:
        make_exp_dirs(opt)
        start_epoch = current_iter = 0

    logger, tb_logger = init_loggers(opt)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = create_train_val_dataloader(opt, logger)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # 数据预取
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode', 'cpu')
    if prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        assert opt['datasets']['train'].get('pin_memory', True), "CUDA prefetch requires pin_memory=True"
    else:
        prefetcher = CPUPrefetcher(train_loader)
    logger.info(f'Using {prefetch_mode} prefetcher')

    # 训练循环
    best_metrics = {}
    start_time = time.time()
    while current_iter <= total_iters:
        if opt['dist']:
            train_sampler.set_epoch(start_epoch)

        prefetcher.reset()
        train_data = prefetcher.next()
        while train_data is not None:
            data_time = time.time()

            # 梯度累积
            accum_steps = opt['train'].get('accumulation_steps', 1)
            if current_iter % accum_steps == 0:
                model.optimizer_g.zero_grad()

            # 学习率更新
            model.update_learning_rate(current_iter, opt['train'].get('warmup_iter', -1))

            # 前向/反向传播
            model.feed_data(train_data)
            model.optimize_parameters(current_iter, tb_logger)
            
            # 梯度裁剪
            if opt['train'].get('use_grad_clip', True):
                torch.nn.utils.clip_grad_norm_(model.net_g.parameters(), 0.01)

            # 梯度累积更新
            if (current_iter + 1) % accum_steps == 0:
                model.optimizer_g.step()

            # 日志记录
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {
                    'epoch': start_epoch,
                    'iter': current_iter,
                    'time': time.time() - data_time,
                }
            # 安全处理学习率
            lrs = model.get_current_learning_rate()
            if isinstance(lrs, (list, tuple)):
                log_vars['lr'] = float(lrs[0])
            else:
                log_vars['lr'] = float(lrs)
    
            # 安全处理模型日志
            current_log = model.get_current_log()
            for k, v in current_log.items():
                if isinstance(v, (list, tuple)):
                    log_vars[k] = float(np.mean(v))
                elif isinstance(v, (torch.Tensor, np.ndarray)):
                    log_vars[k] = float(v.mean().item())
                else:
                    log_vars[k] = v
    
            msg_logger(log_vars)

            if tb_logger:
                for k, v in log_vars.items():
                    try:
                        tb_logger.add_scalar(f'train/{k}', float(v), current_iter)
                    except ValueError as e:
                        logger.warning(f'Skip logging {k}: {str(e)}')

            # 验证
            if opt.get('val') and current_iter % opt['val']['val_freq'] == 0:
                val_log = model.validation(val_loader, current_iter, tb_logger,
                                          opt['val']['save_img'], opt['val'].get('rgb2bgr', True),
                                          opt['val'].get('use_image', True))
                if opt['rank'] == 0 and val_log.get('metrics'):
                    for metric, val in val_log['metrics'].items():
                        if metric not in best_metrics or val > best_metrics[metric]:
                            best_metrics[metric] = val
                            model.save(start_epoch, current_iter, is_best=True)
                            logger.info(f'New best {metric}: {val:.4f}')

            current_iter += 1
            if current_iter > total_iters:
                break
            train_data = prefetcher.next()

        start_epoch += 1

    # 训练结束
    logger.info(f'Training completed in {datetime.timedelta(seconds=int(time.time() - start_time))}')
    model.save(-1, -1)  # 保存最终模型
    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
    main()