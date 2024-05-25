import torch

from kornia.losses import ssim as ssim_

import numpy as np


def huber_loss(pred, target, delta=0.1, reduction='mean'):
    rel = (pred - target).abs()
    sqr = 0.5 / delta * rel * rel
    loss = torch.where(rel > delta, rel - 0.5 * delta, sqr)

    if reduction == 'mean':
        loss = loss.mean()

    return loss


@torch.jit.script
def robust_rgb_loss(rgb_gt,rgb_outputs,patch_size:int):
    # https://arxiv.org/pdf/2302.00833.pdf
    out_patches = rgb_outputs.view(-1,1,patch_size,patch_size,3)
    gt_patches = rgb_gt.view(-1,1,patch_size,patch_size,3)
    device = rgb_outputs.device
    batch_size = out_patches.shape[0]
    residuals = torch.mean((out_patches - gt_patches)**2,dim=-1)
    with torch.no_grad():
        med_residual = torch.quantile(residuals,.9) # 5 article 
        #equation 8
        weight = (residuals<=med_residual).float()#B x patch x patch
        #equation 9
        weight = torch.nn.functional.pad(weight,(1,1,1,1),mode='replicate')
        blurred_w = (torch.nn.functional.conv2d(weight,(1/9.)*torch.ones((1,1,3,3),device=device),
                padding='valid')>=0.5).float()
        #equation 10
        expected_w = blurred_w.view(batch_size,-1).mean(1)
        weight_r8 = (expected_w >= 0.6).float() #Bx1
        #the paper uses the lines below and multiplies residuals by final_w, but I found that just using
        #the value for all 16 pixels worked better.
        # final_w = torch.zeros((batch_size,patch_size,patch_size),device=device)
        # final_w[:,
        #         (patch_size//2-patch_size//4):(patch_size//2+patch_size//4),
        #         (patch_size//2-patch_size//4):(patch_size//2+patch_size//4)] = weight_r8[:,None,None]
    loss = torch.mean(residuals.squeeze()*weight_r8[:,None,None])
    return loss


def uncertainty_aware_loss(inputs, gt_rgb, beta_min=0.05, eta=3):
    
    beta = torch.sum(inputs['weights'] * inputs['uncert']) + beta_min

    term1 = (inputs['image'] - gt_rgb) ** 2
    # robust = 
    color = (term1 / (2 * beta ** 2)).mean()
    logbeta = (eta + torch.log(beta).mean()) / 2  # +3 to make c_b positive since beta_min = 0.05
    
    return torch.add(color, logbeta)

@torch.jit.script
def solar_correction(sun_sc, transparency, weights, lambda_sc:float):
    # computes the solar correction terms defined in Shadow NeRF and adds them to the dictionary of losses

    term2 = torch.sum(torch.square(transparency - sun_sc), -1)

    term3 = 1 - torch.sum( weights * sun_sc, -1)
    sc_term2 = lambda_sc/3. * torch.mean(term2)
    sc_term3 = lambda_sc/3. * torch.mean(term3)
    return sc_term2 + sc_term3


class SatNerfLoss(torch.nn.Module):
    def __init__(self, lambda_sc=0.1, with_beta=True, only_solar=False, no_log=False, batch_size=4096):
        super().__init__()
        self.lambda_sc = lambda_sc
        self.with_beta = with_beta
        self.loss = torch.nn.MSELoss(reduction='none')
        self.only_solar = only_solar
        self.batch_size = batch_size
        self.last = (0,0,0,0)
        self.no_log = no_log


    def forward(self, inputs, targets, after_n_epoch=True, B=1, H=None, W=None, training=True):
              

        loss_dict = {}
        if training:
            
            # MSE+SOLAR
            if self.only_solar:
                loss_dict['image'] = self.loss(inputs['image'], targets).mean(-1)
                if self.lambda_sc > 0:
                    #print("MSE+SOLAR")
                    loss_dict['sc'] = solar_correction(inputs, self.lambda_sc) # .mean(-1) # avec robust
                    # L =                  LRGB(R)        + λ SCLSC(RSC)  
                    loss = torch.add(loss_dict['image'], loss_dict['sc'])
            
            # BETA
            elif self.with_beta:

                # then after N epoch BetaLoss+Solar
                if after_n_epoch:
                    loss_dict['image'] = uncertainty_aware_loss(inputs, targets, beta_min=0.05, eta=3) # BETA loss
                # MSE+solar
                else :
                    if targets.shape[0] == self.batch_size:
                    # print("robust") rgb_gt,rgb_outputs,patch_size
                        image_pred = inputs['image']
                        loss_dict['image'] = robust_rgb_loss(rgb_gt=targets, rgb_outputs=image_pred, patch_size=16)
                    else: 
                        print("targets.shape[0] == self.batch_size Noooooo", targets.shape[0], self.batch_size)
                        loss_dict['image'] = self.loss(inputs['image'], targets).mean(-1)
                   

                if self.lambda_sc > 0:
                    shade=inputs['shade'].squeeze()
                    transparency=inputs['transparency'].detach()
                    weights=inputs['weights'].detach()
                    loss_dict['sc'] = solar_correction(sun_sc=shade, transparency=transparency, weights=weights, lambda_sc=self.lambda_sc) # .mean(-1) # avec robust
                
                # L =                  LRGB(R)        + λ SCLSC(RSC)  
                loss = torch.add(loss_dict['image'], loss_dict['sc'])

            # ROBUST
            else:
                if targets.shape[0] == self.batch_size:
                    # print("robust")
                    image_pred = inputs['image']
                    loss_dict['image'] = robust_rgb_loss(rgb_gt=targets, rgb_outputs=image_pred, patch_size=16)
                else: 
                    # bc the last batch'size processed by the device, we cannot use robust loss patch
                    loss_dict['image'] = self.loss(inputs['image'], targets).mean(-1)

                if self.lambda_sc > 0:
                    shade=inputs['shade'].squeeze()
                    transparency=inputs['transparency'].detach()
                    weights=inputs['weights'].detach()
                    loss_dict['sc'] = solar_correction(sun_sc=shade, transparency=transparency, weights=weights, lambda_sc=self.lambda_sc) # .mean(-1) # avec robust
                
            

                # L =                  LRGB(R)        + λ SCLSC(RSC)  
                loss = torch.add(loss_dict['image'], loss_dict['sc'])
            
            if not self.no_log: 
                if targets.shape[0] == self.batch_size : 
                    
                    sc_, mse_ = loss_dict['sc'].mean(), loss_dict['image'].mean(-1)
                    self.last = sc_, mse_
                    # commenter la deuxième partie pour mesure du temps de train
                    return loss , sc_, mse_
                else:
                    return loss , self.last[0], self.last[1] # for the same purpose as the robust loss 
            else: 
                return loss , None, None, None, None
        else:
            pred_rgb = inputs['image'].reshape(B, H, W, 3)
            loss_dict['image'] = self.loss(pred_rgb, targets).view(-1,3).mean(-1)

            # # L =                  LRGB(R)        + λ SCLSC(RSC)  
            loss = loss_dict['image']
            
            ssim_, psnr_, mse_ = ssim(pred_rgb, targets), psnr(pred_rgb, targets), mse(pred_rgb, targets) 
            return loss, ssim_, psnr_, mse_


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


def ssim(image_pred, image_gt):
    """
    image_pred and image_gt: (1, 3, H, W)
    important: kornia==0.5.3
    """
    return torch.mean(ssim_(image_pred, image_gt, 3))
