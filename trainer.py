import os
import time
import glob
import tqdm
import imageio
import tensorboardX
import numpy as np
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import trimesh
from rich.console import Console
from torch_ema import ExponentialMovingAverage
from torchmetrics.functional import structural_similarity_index_measure

from loss import mse, psnr, ssim, SatNerfLoss
from .sat_utils import create_dsm_mae
from nerf.utils import get_parameters, plot_train_stats, srgb_to_linear, linear_to_srgb, save_output_image, cv2_save_plot, predefined_val_ts, extract_geometry

class Trainer(object):
    def __init__(self,
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network
                 t_embbeding_vocab = 30, # sat nerf
                 t_embbeding_tau=4, # sat nerf
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 gt_dir='',
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):

        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.gt_dir = gt_dir
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        self.ds_drop = np.round(self.opt.ds_drop * self.opt.max_steps)

        self.console = Console()

        self.no_log = opt.no_log

        model.to(self.device)
        if self.world_size > 1:
            print("init world size")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        self.models = {}
        self.models['model'] = self.model

        self.embedding_t = torch.nn.Embedding(t_embbeding_vocab, t_embbeding_tau).to(self.device)
        self.models["t"] = self.embedding_t

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion if criterion is not None else SatNerfLoss()
        self.last = (0, 0, 0)
        
        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1:
            import lpips
            self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        self.parameters = get_parameters(self.models)
        if optimizer is None:
            self.optimizer = optim.Adam(self.parameters, lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        self.render_time = 0
        self.inference_time = 0

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            "sc": [], "ssim": [], "psnr": [], "mse": [], "time" :[], "gpu_consuMB":[],
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

            # plot stats
            self.plots_path = os.path.join(self.workspace, 'metrics')
            os.makedirs(self.plots_path, exist_ok=True)

            # plot stats
            self.gt_path = os.path.join(self.workspace, 'gt')
            os.makedirs(self.gt_path, exist_ok=True)

        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    def train(self, train_loader, valid_loader, max_epochs, save_plot=True):
        self.log(f"\n\n[CONGIG] {self.opt}\n\n")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_train = time.time()
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.evaluate_one_epoch(valid_loader, for_dsm=True) # second time for dsm generation with different coordinate

                self.save_checkpoint(full=True, best=True)

        if save_plot:
            if self.opt.loss == "mse":
                plot_train_stats(self.stats, self.epoch, self.plots_path, only_mse=True)
            else:
                plot_train_stats(self.stats, self.epoch, self.plots_path, only_mse=False)

        training_time = time.time() - start_train
        self.log(f"==> Finished Train in {training_time}")
        self.log(f"==> Total rendering time {training_time, self.inference_time, self.render_time}")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def train_one_epoch(self, loader):
        if not self.no_log:
            self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        start_one_epoch = time.time()

        total_loss = 0
        if not self.no_log:
            total_mse, total_sc = 0, 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        if not self.no_log:
            loss_ite, ssim_ite, psnr_ite, mse_ite, sc_ite, = [], [], [], [], []

        self.local_step = 0

        for data in loader:
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            for param in self.model.parameters():
                param.grad = None

            start_step = time.time()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                if self.opt.loss == "sat":
                    preds, truths, loss, sc, mse_val = self.train_step(data)
                else:
                    preds, truths, loss, mse_val = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()
            self.render_time += (time.time() - start_step)

            loss_val = loss.detach()
            total_loss += loss_val

            if not self.no_log:
                if self.opt.loss == "sat":
                    sc_ite.append(sc.detach())
                    total_sc += sc.detach()

                loss_ite.append(loss_val)
                mse_ite.append(mse_val.detach())
                total_mse += mse_val.detach()

            if not self.no_log:
                if self.local_rank == 0:
                    if self.report_metric_at_train:
                        for metric in self.metrics:
                            metric.update(preds, truths)

                    if self.use_tensorboardX:
                        self.writer.add_scalar("train/loss", loss_val, self.global_step)
                        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                    if self.scheduler_update_every_step:
                        if self.opt.loss == "sat":
                            if self.local_step % 4096 == 0:
                                self.log(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), sc={sc:.4f}, loss_image={mse_val:.4f},  lr={self.optimizer.param_groups[0]['lr']:.6f}")

                            pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), sc={sc:.4f}, mse={mse_val:.4f},  lr={self.optimizer.param_groups[0]['lr']:.6f}")

                        else:
                            if self.local_step % 4096 == 0:
                                self.log(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), loss_image={mse_val:.4f},  lr={self.optimizer.param_groups[0]['lr']:.6f}")

                            pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), mse={mse_val:.4f},  lr={self.optimizer.param_groups[0]['lr']:.6f}")

                    else:
                        pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step

        if not self.no_log:
            self.stats["loss"].append(loss_ite)
            self.stats["ssim"].append(ssim_ite)
            self.stats["psnr"].append(psnr_ite)
            self.stats["mse"].append(mse_ite)
            if self.opt.loss == "sat":
                self.stats["sc"].append(sc_ite)

        end_one_epoch = time.time() - start_one_epoch
        self.stats["time"].append(end_one_epoch)

        if not self.no_log:
            if self.local_rank == 0:
                pbar.close()
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        self.log(metric.report(), style="red")
                        if self.use_tensorboardX:
                            metric.write(self.writer, self.epoch, prefix="train")
                        metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        if not self.no_log:
            if self.opt.loss == "sat":
                self.log(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), sc={sc:.4f}, loss_image={mse_val:.4f},  lr={self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                self.log(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), loss_image={mse_val:.4f},  lr={self.optimizer.param_groups[0]['lr']:.6f}")

            self.log(f"==> Finished Epoch {self.epoch} in {end_one_epoch:.2f}")

    def train_step(self, data):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        nears = data['nears'] # [B, N]
        fars = data['fars'] # [B, N]
        images = data['rgbs'] # [B, N, 3/4]
        C = 3  # r g b

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        else:
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        gt_rgb = images
        # SAT
        if 'ts' in data:
            sun_d = data['sun_d']
            ts = self.models["t"](data['ts'])  

            chunk_size = rays_o.shape[1]
            outputs = defaultdict(list)
            
            rendered_ray_chunks = self.model.render(
                rays_o[0:chunk_size],
                rays_d[0:chunk_size],
                sun_d[0:chunk_size],
                ts[0:chunk_size],
                nears[0:chunk_size],
                fars[0:chunk_size],
                staged=False, bg_color=bg_color, perturb=False, force_all_rays=True, **vars(self.opt))

            for k, v in rendered_ray_chunks.items():
                outputs[k] += [v]
            for k, v in outputs.items():
                outputs[k] = torch.cat(v, 0)

        else:
            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=False, force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))

        pred_rgb = outputs['image']

        # LOSS
        if 'ts' in data:
            if self.opt.loss == "sat":
                after_N_epoch = self.epoch > self.opt.first_beta_epoch
                loss, sc, mse_val = self.criterion(outputs, gt_rgb, after_N_epoch) # .mean(-1) # [B, N, 3] --> [B, N]
            else:
                loss = self.criterion(pred_rgb, gt_rgb).mean(-1)
                if gt_rgb.shape[0] == self.opt.batch_size:
                    H = W = int(np.sqrt(gt_rgb.shape[0]))
                    B = 1
                    pred = pred_rgb.reshape(B, 3, H, W)
                    target = gt_rgb.reshape(B, 3, H, W)
                    self.last = structural_similarity_index_measure(pred, target), loss, -10*torch.log10(loss)
                    ssim_val, mse_val, psnr_val = self.last[0], self.last[1], self.last[2]
                else:
                    ssim_val, mse_val, psnr_val = self.last[0], self.last[1], self.last[2]
        else:
            loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]
        
        loss = loss.mean()

        lambda_opacity = 1e-3
        o = outputs['opacity'] + 1e-10
        outputs['opacity'] = lambda_opacity * (-o * torch.log(o))
        pred_weights_sum = outputs['weights_sum'] + 1e-8
        loss_ws = -1e-1 * pred_weights_sum * torch.log(pred_weights_sum)
        loss = loss + loss_ws.mean()

        if self.opt.loss == "sat":
            return pred_rgb, gt_rgb, loss, sc, mse_val
        else:
            return pred_rgb, gt_rgb, loss, ssim_val, psnr_val.mean(), mse_val.mean()

    @torch.no_grad()
    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    @torch.no_grad()
    def evaluate_one_epoch(self, loader, for_dsm=False, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_uncert_scene, truths, loss, ssim_, psnr_, mse_ = self.eval_step(data)

                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    if for_dsm:
                        preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)]
                        dist.all_gather(preds_depth_list, preds_depth)
                        preds_depth = torch.cat(preds_depth_list, dim=0)
                    else:
                        preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)]
                        dist.all_gather(preds_list, preds)
                        preds = torch.cat(preds_list, dim=0)

                        truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)]
                        dist.all_gather(truths_list, truths)
                        truths = torch.cat(truths_list, dim=0)

                loss_val = loss.detach()
                total_loss += loss_val

                if self.local_rank == 0:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                    if for_dsm:
                        depth_tif_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:02d}_depth.tif')
                        src_path = os.path.join(self.opt.img_dir, data['src_id'] + ".tif")
                        save_output_image(preds_depth[0], depth_tif_path, src_path)
                        save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:02d}_depth.png')
                    else:
                        save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:02d}_rgb.png')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        rgb_tif_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:02d}_rgb.tif')
                        src_path = os.path.join(self.opt.img_dir, data['src_id'] + ".tif")
                        save_output_image(preds[0], rgb_tif_path, src_path)

                        if self.opt.color_space == 'linear':
                            preds = linear_to_srgb(preds)

                        pred = preds[0].detach().cpu().numpy()
                        pred = (pred * 255).astype(np.uint8)
                        cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

                    if for_dsm:
                        pred_depth = preds_depth[0].detach().cpu().numpy()
                        pred_depth = (pred_depth * 255).astype(np.uint8)
                        cv2.imwrite(save_path_depth, pred_depth)
                        save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:01d}_depth_plot.png')
                        mae_, gt_dsm, diff = create_dsm_mae(data, preds_depth, self.epoch, self.workspace, self.gt_dir, name, loader._data, self.local_step)
                        cv2_save_plot(preds_depth[0], save_path_depth, f"{data['src_id'][:-3]} step =  {self.local_step:01d} MAE : {mae_:4f}", is_depth=True)
                    else:
                        save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:01d}_rgb_plot.png')
                        mae_ = 1000
                        cv2_save_plot(pred, save_path, f"{data['src_id'][:-3]} step = {self.local_step:01d} SSIM = {ssim_:.4f} | PSNR = {psnr_:.4f}")

                    if self.epoch > self.opt.first_beta_epoch:
                        uncert_scene_tif_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:02d}_uncert_scene.tif')
                        src_path = os.path.join(self.opt.img_dir, data['src_id'] + ".tif")
                        save_output_image(preds_uncert_scene[0], uncert_scene_tif_path, src_path)
                        pred_uncert_scene = preds_uncert_scene[0].detach().cpu().numpy()
                        pred_uncert_scene = (pred_uncert_scene * 255).astype(np.uint8)
                        save_path_uncert_scene = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:01d}_uncert_scene_plot.png')
                        cv2_save_plot(pred_uncert_scene, save_path_uncert_scene, f"{data['src_id'][:-3]} step =  {self.local_step:01d} MAE : {mae_:4f}")

                    if self.epoch == 1:
                        truth = truths[0].detach().cpu().numpy()
                        truth = (truth * 255).astype(np.uint8)
                        cv2_save_plot(truth, self.gt_path+'/'+data['src_id']+"_gt.png", data['src_id'])

                        if for_dsm:
                            plt.imshow(gt_dsm)
                            plt.title("gt dsm "+data['src_id'])
                            plt.savefig(self.gt_path+'/'+data['src_id'][:-3]+"_gt_dsm.png", bbox_inches='tight')
                            plt.clf()
                    if for_dsm:
                        plt.imshow(diff, cmap='seismic')
                        plt.title("diff dsm "+data['src_id'])
                        plt.savefig(self.gt_path+'/'+data['src_id'][:-3]+"_"+str(self.local_step)+"_diff_dsm.png", bbox_inches='tight')
                        plt.clf()

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result)
            else:
                self.stats["results"].append(average_loss)

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Metrics epoch {self.epoch} : loss = {loss_val:.4f} | SSIM = {ssim_:.4f} | PSNR = {psnr_:.4f} | MSE = {mse_:.4f} | MAE = {mae_:4f}")
        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    @torch.no_grad()
    def eval_step(self, data):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        nears = data['nears'] # [B, N]
        fars = data['fars'] # [B, N]
        images = data['rgbs'] # [B, H, W, 3/4]

        B, H, W, C =  1, data['H'], data['W'], 3 # images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        # SAT
        if 'ts' in data:
            sun_d = data['sun_d']
            t = predefined_val_ts(data["src_id"])

            ts = self.models["t"](t * torch.ones(rays_o.shape[1], 1).long().cuda())

            outputs = self.model.render(rays_o, rays_d, sun_d, ts, nears, fars, staged=False, bg_color=bg_color, perturb=False, **vars(self.opt))

            pred_rgb = outputs['image'].reshape(B, H, W, 3)
            pred_depth = outputs['depth'].reshape(B, H, W)
            pred_uncert_scene = outputs['uncert_scene'].reshape(B, H, W)

            if self.opt.loss == "sat":
                loss, ssim_, psnr_, mse_ = self.criterion(outputs, gt_rgb, after_n_epoch=True, B=B, H=H, W=W, training=False)
                loss = loss.mean()
            else:
                loss = self.criterion(pred_rgb, gt_rgb).mean()
                with torch.no_grad():
                    ssim_, psnr_, mse_ = ssim(pred_rgb, gt_rgb), psnr(pred_rgb, gt_rgb), mse(pred_rgb, gt_rgb)
        else:
            outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

            pred_rgb = outputs['image'].reshape(B, H, W, 3)
            pred_depth = outputs['depth'].reshape(B, H, W)

            loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, pred_uncert_scene, gt_rgb, loss, ssim_, psnr_, mse_

    @torch.no_grad()
    def test(self, loader, save_path=None, name=None, write_video=True):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
        start_infe_test = time.time()
        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
        print("time infe_test", time.time() - start_infe_test)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")

    @torch.no_grad()
    def test_step(self, data, bg_color=1, perturb=False):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        nears = data['nears'] # [B, N]
        fars = data['fars'] # [B, N]
        H, W = data['H'], data['W'] # images.shape

        if 'ts' in data:
            sun_d = data['sun_d']
            t = predefined_val_ts(data["src_id"])

            ts = self.models["t"](t * torch.ones(rays_o.shape[1], 1).long().cuda().squeeze())

            outputs = self.model.render(rays_o, rays_d, sun_d, ts, nears, fars, staged=False, bg_color=bg_color, perturb=perturb, **vars(self.opt))
        else:
            outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):
        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:
            state['model'] = self.model.state_dict()
            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)
        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

    @torch.no_grad()
    def save_mesh(self, save_path=None, resolution=256, threshold=10):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        mesh.export(save_path)

        save_obj = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.obj')
        obj = trimesh.exchange.obj.export_obj(mesh)
        with open(save_obj, "w+") as file:
            file.write(obj)

        self.log(f"==> Finished saving mesh.")
