import torch
import argparse
import numpy as np
from nerf.provider import SATNeRFDataset
from nerf.utils import PSNRMeter, SSIMMeter
from nerf.trainer import Trainer 

from functools import partial
from loss import huber_loss, SatNerfLoss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--gt_dir', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--beta_1', type=float, default=0.9, help="initial beta_1")
    parser.add_argument('--beta_2', type=float, default=0.99, help="initial beta_2")
    parser.add_argument('--weight_decay', type=float, default=1e-6, help="initial weight_decay")
    parser.add_argument('--ckpt', type=str, default='scratch')
    parser.add_argument('--num_rays', type=int, default=1024, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")

    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch size of rays")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--mesh_res', type=int, default=256, help='')


    ### params to set for the report
    parser.add_argument('--optim', type=str, default='adam', help="adam, rprop")
    parser.add_argument('--encoding', type=str, default='hashgrid', help="frequency, hashgrid, mapping")
    parser.add_argument('--encoding_dir', type=str, default='sphere_harmonics', help="sphere_harmonics, frequency, mapping")
    parser.add_argument('--siren', action='store_true', help="Siren in MLP")
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers')
    parser.add_argument('--hidden_dim', type=int, default=128, help='number of neurons')
    parser.add_argument('--skips', type=int, default=2, help='not well understood ...')
    parser.add_argument('--max_steps', type=int, default=128*1, help="max num steps sampled per ray (only valid when using --cuda_ray)")

    # see https://pytorch.org/docs/stable/nn.init.html
    parser.add_argument('--init_weight', type=str, default='U', help="U Xu Xn Ku Kn N O ")
    parser.add_argument('--nonlinearity', type=str, help="leaky_relu relu")
    parser.add_argument('--gain_nonlinearity', type=str, help="relu leaky_relu tanh")
    parser.add_argument('--mode', type=str, help="fan_in fan_out")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.0001, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=20, help="threshold for density grid to be occupied")
    parser.add_argument('--density_scale', type=int, default=1, help="multiply sigma after forward")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    # SAT
    parser.add_argument('--root_dir', type=str,  default=None, help='root directory of the input dataset')
    parser.add_argument('--img_dir', type=str, default=None,  help='Directory where the images are located (if different than root_dir)')
    parser.add_argument('--cache_dir', type=str, default=None, help='directory where cache for the current dataset is found')
    parser.add_argument('--downscale', type=int, default=1, help='Downsample the input images by a factor 1/n')
    parser.add_argument('--sat', action='store_true', help="use SAT NeRF MLP")
    parser.add_argument('--with_beta', action='store_true', help="add beta loss")
    parser.add_argument('--only_solar', action='store_true', help="only mse + solar loss")
    parser.add_argument('--loss',  type=str, default="mse", help="spec the loss (mse, solar correc, uncertainiy, ...")

     # other sat-nerf specific stuff
    parser.add_argument('--ds_lambda', type=float, default=1000/3,
                        help='float that multiplies the depth supervision auxiliary loss')
    parser.add_argument('--ds_drop', type=float, default=0.25,
                        help='portion of training steps at which the depth supervision loss will be dropped')
    parser.add_argument('--ds_noweights', action='store_true',
                        help='do not use reprojection errors to weight depth supervision loss')


    parser.add_argument('--first_beta_epoch', type=int, default=1,help='')
    parser.add_argument('--t_embbeding_vocab', type=int, default=30,help='')
    parser.add_argument('--lambda_sc', type=float, default=0.05, help="float that multiplies the solar correction auxiliary loss")

    parser.add_argument('--no_log', action='store_true', help="no log for speed training test")


    # Encodings
    parser.add_argument('--grid_size', type=int, default=128, help='grid_size hashgrid')
    parser.add_argument('--num_levels', type=int, default=16, help='num_levels hashgrid')
    parser.add_argument('--desired_resolution', type=int, default=0, help='desired_resolution hashgrid')
    parser.add_argument('--level_dim', type=int, default=2, help='level_dim hashgrid')
    parser.add_argument('--base_resolution', type=int, default=16, help='base_resolution hashgrid')
    parser.add_argument('--degree', type=int, default=4,)
    parser.add_argument('--log2_hashmap_size', type=int, default=19, help='log2_hashmap_size hashgrid')


    opt = parser.parse_args()
    use_tensorboardX = False
    if opt.O:
        opt.cuda_ray = True
        opt.preload = True

    opt.fp16 = True

    from nerf.network_sat import NeRFNetwork


    eval_interval = 1 # pour ne pas avoir d'eval pour le test time 1000
  
    model = NeRFNetwork(
        encoding=opt.encoding,
        encoding_dir=opt.encoding_dir,
        siren=opt.siren,
        num_layers=opt.num_layers,
        hidden_dim=opt.hidden_dim,
        skips=opt.skips,

        num_levels=opt.num_levels,
        desired_resolution=opt.desired_resolution,
        level_dim=opt.level_dim,
        base_resolution=opt.base_resolution,
        log2_hashmap_size=opt.log2_hashmap_size,

        init_weight=opt.init_weight,  # U Xu Xn Ku Kn N O
        nonlinearity=opt.nonlinearity, # leaky_relu relu
        gain_nonlinearity=opt.gain_nonlinearity,  #  relu leaky_relu tanh
        mode=opt.mode, # fan_in fan_out

        degree=opt.degree,

        bound=opt.bound,
        grid_size=opt.grid_size,
        cuda_ray=opt.cuda_ray,
        density_scale=opt.density_scale, # 1
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        t_embedding_dims=4 # if changed then modify ts var of kernel_march_rays_train func in cuda backend
    )

    downscale=opt.downscale
    if opt.loss == "sat":
        criterion = SatNerfLoss(lambda_sc=opt.lambda_sc, with_beta=opt.with_beta, only_solar=opt.only_solar, no_log=opt.no_log, batch_size=opt.batch_size)
    elif opt.loss == "part_huber":
        criterion = partial(huber_loss, reduction='none')
    elif opt.loss == "huber":
        criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?
    else:
        criterion = torch.nn.MSELoss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:

        metrics = [PSNRMeter(), SSIMMeter()]
        trainer = Trainer('ngp', opt, model, use_tensorboardX=use_tensorboardX, 
                          device=device, gt_dir=opt.gt_dir, workspace=opt.workspace, criterion=criterion, 
                          fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.root_dir is not None:
            test_loader = SATNeRFDataset(opt, device=device, type='test', downscale=downscale).dataloader()
        else:
            print("!!!!!!! opt.root_dir is None  !!!!!!!!!!")

        if test_loader.has_gt:
            trainer.evaluate(test_loader) # blender has gt, so evaluate it.

        trainer.test(test_loader, write_video=False) # test and save video

        trainer.save_mesh(resolution=opt.mesh_res, threshold=opt.density_thresh)

    else:

        if opt.optim == "adam":
            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), lr=opt.lr, betas=(opt.beta_1, opt.beta_2), eps=1e-15, weight_decay=opt.weight_decay, fused=True)
        elif opt.optim == "rmsprop":
            optimizer = lambda model: torch.optim.RMSprop(model.get_params(opt.lr), lr=opt.lr, alpha=0.99, eps=1e-15, weight_decay=1e-6, momentum=0,
                                                                            centered=False, foreach=None, maximize=False, differentiable=False)
        elif opt.optim == "radam":
            optimizer = lambda model: torch.optim.RAdam(model.get_params(opt.lr), lr=opt.lr, betas=(opt.beta_1, opt.beta_2), eps=1e-15, weight_decay=opt.weight_decay, foreach=None, differentiable=False)


        if opt.root_dir is not None:

            dataSAT = SATNeRFDataset(opt, device=device, type='train', downscale=downscale)

            boundary_rayo_norma = dataSAT.ray_ox_min, dataSAT.ray_ox_max, dataSAT.ray_oy_min, dataSAT.ray_oy_max

            train_loader = dataSAT.dataloader()
        else:
            print("!!!!!!! opt.root_dir is None  !!!!!!!!!!")

        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), SSIMMeter()]
        trainer = Trainer('ngp', opt, model, use_tensorboardX=use_tensorboardX, device=device, gt_dir=opt.gt_dir, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=eval_interval)


        if opt.root_dir is not None:
            valid_loader = SATNeRFDataset(opt, device=device, type='val', downscale=downscale, boundary_rayo_norma=boundary_rayo_norma).dataloader()
        else:
            print("!!!!!!! opt.root_dir is None  !!!!!!!!!!")


        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        print("Nb epochs ", max_epoch)
        trainer.train(train_loader, valid_loader, max_epoch)

        # also test
        if opt.root_dir is not None:
            test_loader = SATNeRFDataset(opt, device=device, type='test', downscale=downscale, boundary_rayo_norma=boundary_rayo_norma).dataloader()
        else:
            print("!!!!!!! opt.root_dir is None  !!!!!!!!!!")

        if test_loader.has_gt:
            trainer.evaluate(valid_loader) # blender has gt, so evaluate it.
           
        trainer.test(test_loader, write_video=False) # test and save video

        trainer.save_mesh(resolution=opt.mesh_res, threshold=opt.density_thresh)
