import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.models.losses.losses import SSIMLoss

from basicsr.models.archs.v51_fft_arch import DenoisingModule #去噪模块
from torch.cuda.amp import GradScaler, autocast


loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class ImageFftModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageFftModel, self).__init__(opt)
        
        # # 初始化去噪模块
        # self.denoising_module = DenoisingModule().to(self.device)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
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
        train_opt = self.opt['train']
            
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fft_loss_opt'):
            fft_type = train_opt['fft_loss_opt'].pop('type')
            cri_fft_cls = getattr(loss_module, fft_type)
            self.cri_fft = cri_fft_cls(**train_opt['fft_loss_opt']).to(
                self.device)
        else:
            self.cri_fft = None

        # if train_opt.get('perceptual_loss_opt'):
        #     from basicsr.models.losses import PerceptualLoss
        #     self.cri_perceptual = PerceptualLoss(
        #         layer_weights=train_opt['perceptual_loss_opt'].get('layer_weights', {'3': 1.0, '8': 1.0, '15': 1.0, '22': 1.0}),
        #         use_l1_loss=train_opt['perceptual_loss_opt'].get('use_l1_loss', False)
        #     ).to(self.device)
        # else:
        #     self.cri_perceptual = None

        if self.cri_pix is None and self.cri_fft is None:
            raise ValueError('All losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                 **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supported yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device, non_blocking=True)  # 使用 non_blocking=True 提高效率
        
        # with torch.no_grad():
        #     self.lq = self.denoising_module(self.lq)  # 使用去噪模块进行预处理
            
        if 'gt' in data:
            self.gt = data['gt'].to(self.device,non_blocking=True)

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

        parts = []
        idxes = []

        i = 0  # 0~h-1
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
                    self.lq[:, :, i // self.scale:(i + crop_size_h) // self.scale, j // self.scale:(j + crop_size_w) // self.scale])
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

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()
        self.accumulation_steps = 32  # 梯度累积步数
        scaler = GradScaler()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()

        # 支持无监督学习
        if self.opt['train'].get('unsupervised', False):
            preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]

            self.output = preds[-1]

            l_total = 0
            loss_dict = OrderedDict()
            # 自监督损失（例如一致性损失）
            if hasattr(self, 'cri_consistency'):
                l_consistency = self.cri_consistency(preds[-1], self.lq)
                l_total += l_consistency
                loss_dict['l_consistency'] = l_consistency
            
            l_total = l_total / self.accumulation_steps  # 平均损失
            scaler.scale(l_total).backward()

        else:
            with autocast():   
                preds = self.net_g(self.lq)
                if not isinstance(preds, list):
                    preds = [preds]

                self.output = preds[-1]

                l_total = 0
                loss_dict = OrderedDict()
        
            # 如果有定义 cri_ssim 并且不为空，则计算 SSIM 损失
            # if self.cri_ssim is not None:
            #     l_ssim = self.ssim_weight * self.cri_ssim(self.output, self.gt)
            #     l_total += l_ssim
            #     loss_dict['l_ssim'] = l_ssim.item()
        
            # 像素损失
            if self.cri_pix:
                l_pix = 0.
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            # FFT 损失
            if self.cri_fft:
                l_fft = self.cri_fft(preds[-1], self.gt)
                l_total += l_fft
                loss_dict['l_fft'] = l_fft

            # 感知损失
            # if self.cri_perceptual:
            #     l_perceptual = self.cri_perceptual(preds[-1], self.gt)
            #     l_total += l_perceptual
            #     loss_dict['l_perceptual'] = l_perceptual
                
            l_total = l_total / self.accumulation_steps  # 平均损失
            l_total.backward()
            
            # 释放不必要的张量
            del preds
            torch.cuda.empty_cache()
            
        if (current_iter + 1) % self.accumulation_steps == 0:
            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()
            scaler.update()
            self.optimizer_g.zero_grad()

        self.log_dict = self.reduce_loss_dict(loss_dict)

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
                b, c, h, w = self.lq[i:j].shape
                h_n = (32 - h % 32) % 32
                w_n = (32 - w % 32) % 32
                in_tensor = F.pad(self.lq[i:j], (0, w_n, 0, h_n), mode='reflect')
                pred = self.net_g(in_tensor)
                pred = pred[:, :, :h, :w]
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        
        # 释放不必要的张量
        del outs
        torch.cuda.empty_cache()
        
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
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

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
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
                # calculate metrics
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

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

        keys = []
        metrics = []
        for name, value in collected_metrics.items():
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

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        for name, value in metric_dict.items():
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