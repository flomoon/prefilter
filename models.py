import math
import logging
import threading
import queue
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import compressai.zoo as ca_zoo
from compressai.layers import GDN
from compressai.models.utils import deconv
from pytorch_msssim import MS_SSIM

from tqdm import tqdm

from codec_ops import batch_codec_fn
from checkpoint import Checkpoint
from utils import parse_target, create_optimizer


###########################################
# 1. Resize (256, 256)
# 2. Filter
# 3. Surrogate codec
# 4. RandomCrop/CenterCrop (224, 224) 
# 5. ResNet50


# 기존에 augmentation은 aspect ratio변화 없이
# 256 x ??? 또는 ??? x 256 로 리사이즈 후
# 크롭..
###########################################


class FilterNet(nn.Module):
    def __init__(self, quality):
        super(FilterNet, self).__init__()
        assert quality > 0 and quality < 9
        self.surrogate_encoder = SurrogateEncoder(quality)

        # 'C' is 320 in paper, they might train the model from the scratch.
        if quality < 5:
            self.pixelwise_estimator = nn.Sequential(
                deconv(192, 128, kernel_size=5, stride=2),
                GDN(128, inverse=True),
                deconv(128, 64, kernel_size=5, stride=2),
                GDN(64, inverse=True),
                deconv(64, 32, kernel_size=5, stride=2),
                GDN(32, inverse=True),
                deconv(32, 16, kernel_size=5, stride=2),
            )
        else:
            self.pixelwise_estimator = nn.Sequential(
                deconv(320, 192, kernel_size=5, stride=2),
                GDN(192, inverse=True),
                deconv(192, 192, kernel_size=5, stride=2),
                GDN(192, inverse=True),
                deconv(192, 192, kernel_size=5, stride=2),
                GDN(192, inverse=True),
                deconv(192, 16, kernel_size=5, stride=2),
            )
        self.filter = nn.Sequential(
            FilterBlock(19, 64),
            FilterBlock(64, 64),
            FilterBlock(64, 64),
            FilterBlock(64, 64),
            # nn.Conv2d(64, 3, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(64, 3, kernel_size=1, stride=1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.surrogate_encoder(x)
        out = self.pixelwise_estimator(out)
        out = torch.cat([x, out], axis=1)
        out = self.filter(out)
        out = out + x
        out = torch.clip(out, 0., 1.)
        return out


class SurrogateEncoder(nn.Module):
    def __init__(self, quality):
        super(SurrogateEncoder, self).__init__()
        self.surrogate = ca_zoo.mbt2018(quality, pretrained=True)
        del self.surrogate.g_s
        for param in self.surrogate.parameters():
            param.requires_grad = False

    def forward(self, x):
        y = self.surrogate.g_a(x)
        z = self.surrogate.h_a(y)
        z_hat, _ = self.surrogate.entropy_bottleneck(z)
        params = self.surrogate.h_s(z_hat)

        y_hat = self.surrogate.gaussian_conditional.quantize(y, "dequantize")
        ctx_params = self.surrogate.context_prediction(y_hat)
        gaussian_params = self.surrogate.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.surrogate.gaussian_conditional(y, scales_hat, means=means_hat)
        return y_likelihoods


class FilterBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FilterBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
        )
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.proj:
            x = self.proj(x)
        out = out + x
        out = self.relu(out)
        return out


class VisionNet(nn.Module):
    def __init__(self, target):
        super(VisionNet, self).__init__()
        target_dict = parse_target(target)

        self.module = target_dict['module']
        for param in self.module.parameters():
            param.requires_grad = False

        cropping_size = target_dict['cropping_size']
        self.random_crop = transforms.RandomCrop([cropping_size, cropping_size])
        self.center_crop = transforms.CenterCrop([cropping_size, cropping_size])
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def forward(self, x, y=None):        
        if isinstance(self.module, MS_SSIM):
            # Here, 'y' is raw 'x'.
            msssim = self.module(x, y)
            return msssim
        else:
            if self.training:
                x = self.random_crop(x)
            else:
                x = self.center_crop(x)
            x = self.normalize(x)
            y_pred = self.module(x)
            if y is None:
                return y_pred
            y_pred = torch.argmax(y_pred, dim=1)
            accuracy = (y_pred == y).sum() / y.size(0)
            return accuracy


class Trainer:
    def __init__(self, config, session_name):
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO)
        logging.info(f"Trainer is created for '{session_name}'.")
        self.config = config

        self.filter_net = FilterNet(self.config.quality)
        self.surrogate_net = ca_zoo.mbt2018(quality=self.config.quality, pretrained=True)
        for param in self.surrogate_net.parameters():
            param.requires_grad = False
        self.vision_net = VisionNet(self.config.target)
        self.filter_net.train()
        self.vision_net.train()
        self.surrogate_net.train()

        if not isinstance(self.vision_net.module, MS_SSIM):
            self.criterion = nn.CrossEntropyLoss()

        trainable_parameters = (
                list(self.filter_net.filter.parameters())
                + list(self.filter_net.pixelwise_estimator.parameters()))
        self.optimizer = create_optimizer(self.config.optimizer, self.config.learning_rate, trainable_parameters)

        output_dir = Path(config.output) / session_name
        self.checkpoint = Checkpoint(output_dir)
        self.writer = SummaryWriter(output_dir)
        self._step = self.checkpoint.resume(self.filter_net, self.optimizer)
        self.cuda()
        self.data_q = queue.Queue(maxsize=4)
        self.run()

    def __call__(self, x):
        out = self.filter_net(x)
        outs = self.surrogate_net(out)
        x_hat = outs['x_hat']
        bpp = self.compute_bpp(outs)
        if isinstance(self.vision_net.module, MS_SSIM):
            out = self.vision_net(x_hat, x)
        else:
            out = self.vision_net(x_hat)
        return out, bpp

    def get_step(self):
        return self._step

    def cuda(self):
        self.filter_net = self.filter_net.cuda()
        self.vision_net = self.vision_net.cuda()
        self.surrogate_net = self.surrogate_net.cuda()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        logging.info("Trainer is loaded on the GPU.")

    def compute_bpp(self, outs):
        size = outs['x_hat'].size()
        num_pixels = size[0] * size[2] * size[3]
        return sum(-torch.log2(likelihoods).sum() / num_pixels
                for likelihoods in outs['likelihoods'].values())

    def put_data(self, data):
        if data:
            data = map(lambda x: x.cuda(), data)
        self.data_q.put(data)
    
    def train_step(self, data):
        x, y = data
        self.optimizer.zero_grad()

        if isinstance(self.vision_net.module, MS_SSIM):
            msssim, loss_bpp = self(x)
            loss_task = 1 - msssim # or -log?
        else:
            y_pred, loss_bpp = self(x)
            loss_task = self.criterion(y_pred, y)

        loss_rd = loss_bpp + self.config.lmbda * loss_task
        loss_rd.backward()
        self.optimizer.step()
        step = self._step + 1

        if step % 100 == 0:
            loss_bpp, loss_task = loss_bpp.item(), loss_task.item()
            if isinstance(self.vision_net.module, MS_SSIM):
                metric = msssim
            else:
                total = y.size(0)
                y_pred = torch.argmax(y_pred, dim=1)
                correct = (y_pred == y).sum().item()
                metric = correct / total
            
            self.writer.add_scalar(f'{self.config.target}/train', metric, step)
            self.writer.add_scalar('loss_bpp/train', loss_bpp, step)
            self.writer.add_scalar('loss_task/train', loss_task, step)
            self.writer.add_scalar('lr/train', self.optimizer.param_groups[0]['lr'], step)
            logging.info(f'Step: {step} | {self.config.target}: {metric:.4f} | '
                         f'Losses - task: {loss_task:.4f} - bpp: {loss_bpp:.4f}')

        if step % self.config.temp_checkpoint_period == 0:
            self.checkpoint.save(self.filter_net, self.optimizer, step)
        self._step = step  # update `self._step` end of the method.
    
    def _run(self):
        while True:
            data = self.data_q.get()
            if data is None:
                break
            self.train_step(data)

    def run(self):
        running_thread = threading.Thread(target=self._run, daemon=True)
        running_thread.start()


class Evaluator:
    def __init__(self, config, checkpoint_path):
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO)
        self.config = config

        if 'base' in checkpoint_path.name:
            self.is_baseline = True
        else:
            self.is_baseline = False
            filter_quality = int(checkpoint_path.name.split('_')[1][1:])
            self.filter_net = FilterNet(filter_quality)
            self.filter_net.eval()

        logging.info(f"Evaluator is created for ['{self.config.eval_target}', '{self.config.eval_codec}'] (session: '{checkpoint_path.name}').")
        self.vision_net = VisionNet(self.config.eval_target)
        self.checkpoint = Checkpoint(checkpoint_path)

        self.vision_net.eval()
        self.data_q = queue.Queue(maxsize=10)
        self.output_q = queue.Queue()

    def __call__(self, x, y=None):

        if self.is_baseline:
            out = x
        else:
            out = self.filter_net(x)

        if self.config.eval_codec == 'surr':
            outs = self.surrogate_net(x)
            x_hat = outs['x_hat']
            bpp = self.compute_bpp(outs)
        elif self.config.eval_codec in ['jpeg', 'webp', 'vtm']:
            device = out.device
            outs = batch_codec_fn(out.detach().cpu().numpy(), self.config.eval_codec, self.quality)
            x_hat = torch.as_tensor(outs['x_hat'], device=device)
            bpp = torch.as_tensor(outs['bpp'].mean(), device=device)
        else:
            raise NotImplemented("'jpeg', 'webp', 'vtm', and 'surroage' are supported.")

        if isinstance(self.vision_net.module, MS_SSIM):
            out = self.vision_net(x_hat, x)
        else:
            out = self.vision_net(x_hat, y)
        return out, bpp

    def cuda(self):
        if not self.is_baseline:
            self.filter_net = self.filter_net.cuda()
        self.vision_net = self.vision_net.cuda()

    def load_weight(self, step=-1):
        self._step = step
        if not self.is_baseline:
            self.checkpoint.load(self.filter_net, step=step)
        self.cuda()
        logging.info(f"Weight (step: {step}) is loaded on GPU.")

    def get_step(self):
        return self._step

    def compute_bpp(self, outs):
        size = outs['x_hat'].size()
        num_pixels = size[0] * size[2] * size[3]
        return sum(-torch.log2(likelihoods).sum() / num_pixels
                for likelihoods in outs['likelihoods'].values())

    def set_quality(self, quality):
        if self.config.eval_codec == 'surr':
            assert 1 <= quality and quality <= 8
            self.surrogate_net = ca_zoo.mbt2018(quality=quality, pretrained=True)
            self.surrogate_net.eval()
            self.surrogate_net.cuda()
        else:
            assert 1 <= quality
        self.quality = quality
        self.metrics = []
        self.bpps = []
        logging.info(f"Codec quality is set to '{self.quality}'.")
    
    def get_output(self):
        metric, bpp = self.output_q.get()
        return metric, bpp, self.quality

    def put_data(self, data):
        if data:
            data = map(lambda x: x.cuda(), data)
        self.data_q.put(data)

    def eval_step(self, data):
        with torch.no_grad():
            x, y = data
            if isinstance(self.vision_net.module, MS_SSIM):
                metric, bpp = self(x)
            else:
                metric, bpp = self(x, y)
        return metric.item(), bpp.item()

    def _run(self):
        metrics, bpps = [], []

        if self.config.eval_codec == 'vtm':
            num_batches = math.ceil(self.config.num_eval_data / self.config.batch_size)
            pbar = tqdm(total=num_batches)

        while True:
            data = self.data_q.get()
            if data is None:
                break # this loop has to be ended with this break.
            metric, bpp = self.eval_step(data)
            metrics.append(metric)
            bpps.append(bpp)

            mean_metric = sum(metrics) / len(metrics)
            mean_bpp = sum(bpps) / len(bpps)

            if self.config.eval_codec == 'vtm':
                pbar.update(1)
                pbar.set_postfix(
                    metric=mean_metric, bpp=mean_bpp)
                
            
        mean_metric = sum(metrics) / len(metrics)
        mean_bpp = sum(bpps) / len(bpps)
        self.output_q.put((mean_metric, mean_bpp))

    def run(self):
        running_thread = threading.Thread(target=self._run, daemon=True)
        running_thread.start()
            


