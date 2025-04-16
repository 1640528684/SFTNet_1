import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import logging

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # 定义网络
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.net_g = self.net_g.to(self.device)  # 确保模型在正确的设备上

        # 加载预训练模型
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        self.train_opt = self.opt['train']

        # 定义像素损失，默认使用 L1Loss
        if self.train_opt.get('pixel_opt'):
            pixel_type = self.train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**self.train_opt['pixel_opt']).to(self.device)
        else:
            # 如果没有定义 pixel_opt，则使用默认的 L1Loss
            from basicsr.models.losses import L1Loss
            self.cri_pix = L1Loss(loss_weight=1.0).to(self.device)

        # 定义感知损失，默认不启用
        if self.train_opt.get('perceptual_opt'):
            percep_type = self.train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **self.train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # 检查是否至少有一个损失函数被定义
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # 设置优化器和学习率调度器
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        # 定义有效的优化器参数
        valid_params_dict = {
            'Adam': ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad'],
            'SGD': ['lr', 'momentum', 'dampening', 'weight_decay', 'nesterov'],
            'AdamW': ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad']
        }
        # 获取当前优化器支持的有效参数列表
        valid_params_list = valid_params_dict.get(optim_type, [])
        # 定义优化器支持的参数
        valid_params = {
            k: v for k, v in train_opt['optim_g'].items()
            if k in valid_params_list  # ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad']
        }

        # 打印传递给优化器的参数（调试信息）
        print(f"Parameters passed to optimizer: {valid_params}")

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}], **valid_params)
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params, **valid_params)
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}], **valid_params)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')

        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        if 'lq' not in data:
            raise KeyError("The 'lq' key is missing in the data dictionary.")
        self.lq = data['lq'].to(self.device)
        if torch.isnan(self.lq).any() or torch.isinf(self.lq).any():
            raise ValueError("Input data 'lq' contains NaN or Inf values.")
    
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            if torch.isnan(self.gt).any() or torch.isinf(self.gt).any():
                raise ValueError("Input data 'gt' contains NaN or Inf values.")

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i // scale * scale
        step_j = step_j // scale * scale

        parts = []
        idxes = []

        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(
                    self.lq[:, :, i // scale:(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def get_current_learning_rate(self):
        return self.optimizer_g.param_groups[0]['lr']

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if hasattr(self, 'train_opt') and self.train_opt.get('mixup', False):
            self.mixup_aug()

        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # 像素损失
        l_pix = 0.
        for pred in preds:
            l_pix_part = self.cri_pix(pred, self.gt)
            if torch.isnan(l_pix_part).any():
                print(f"NaN detected in l_pix_part: pred={pred}, gt={self.gt}")
            l_pix += l_pix_part
        
        if torch.isnan(l_pix).any():
            print(f"NaN detected in l_pix: preds={preds}, gt={self.gt}")
        l_total += l_pix
        loss_dict['l_pix'] = l_pix

        # 感知损失
        l_percep = None
        l_style = None
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())
    
        if torch.isnan(l_total).any():
            print(f"NaN detected in l_total: l_pix={l_pix}, l_percep={l_percep}, l_style={l_style}")

        l_total.backward()

        if hasattr(self, 'train_opt') and 'clip_grad_norm' in self.train_opt and self.train_opt['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.net_g.parameters(),
                max_norm=self.train_opt['clip_grad_norm'],
                norm_type=2.0
            )

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        # 添加学习率到 log_dict
        self.log_dict['lrs'] = [self.get_current_learning_rate()]  # 确保是列表

    def reduce_loss_dict(self, loss_dict):
        """Reduce loss dict.

        In distributed training, it averages the losses among different GPUs.

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        reduced_loss_dict = {}
        with torch.no_grad():
            if self.opt['num_gpu'] > 1:
                for key in loss_dict:
                    reduced_loss_dict[key] = loss_dict[key].mean().item()
            else:
                for key in loss_dict:
                    reduced_loss_dict[key] = loss_dict[key].item()

        # 添加学习率
        reduced_loss_dict['lrs'] = [self.get_current_learning_rate()]  # 确保是列表

        return reduced_loss_dict

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if not torch.distributed.is_initialized():
            self._run_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
            return
        if isinstance(dataloader.dataset, torch.utils.data.Subset):
            dataset_name = dataloader.dataset.dataset.opt['name']
        else:
            dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')
                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                    img_name,
                                                    f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')

        if rank == 0:
            pbar.close()

        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        self._run_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)

    def _run_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if isinstance(dataloader.dataset, torch.utils.data.Subset):
            dataset_name = dataloader.dataset.dataset.opt['name']
        else:
            dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')
                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                    img_name,
                                                    f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        if with_metrics:
            metrics_dict = {}
            for metric in self.metric_results.keys():
                metrics_dict[metric] = self.metric_results[metric] / cnt
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger, metrics_dict)

    def _log_validation_metric_values(self, current_iter, dataset_name,tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.setLevel(logging.INFO)
        logger.info(log_str)
        
        # 记录日志信息到 TensorBoard
        if tb_logger:
            for metric, value in metric_dict.items():
                tb_logger.add_scalar(f'{dataset_name}/m_{metric}', value, current_iter)

        log_dict = OrderedDict()
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
        
    # 添加 get_state_dict 方法
    def get_state_dict(self):
        return self.net_g.state_dict()
    def get_optimizers(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]
    def get_schedulers(self):
        return [scheduler.state_dict() for scheduler in self.schedulers]