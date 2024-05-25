import os
import random
import rasterio
import numpy as np

import cv2
import matplotlib.pyplot as plt

import torch
import mcubes

from packaging import version as pver
from torchmetrics.functional import structural_similarity_index_measure



def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch..html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def visualize_depth(depth):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = np.clip(x, 0, 255)
    return x_

def save_output_image(input, output_path, source_path):
    """
    input: (D, H, W) where D is the number of channels (3 for rgb, 1 for grayscale)
           can be a pytorch tensor or a numpy array
    """
    # convert input to numpy array float32
    if len(input.shape) == 3: # rgb
        H, W, D = input.shape
        input = input.view(D, H, W)
    else: # maybe depth
        H, W = input.shape
        input = input.view(1, H, W)
    if torch.is_tensor(input):
        im_np = input.type(torch.FloatTensor).cpu().numpy()
    else:
        im_np = input.astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(source_path, 'r') as src:
        profile = src.profile
        profile["dtype"] = rasterio.float32
        profile["height"] = im_np.shape[1]
        profile["width"] = im_np.shape[2]
        profile["count"] = im_np.shape[0]
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(im_np)

def cv2_save_plot(x, path, title, is_depth=False):
    # x: [3, H, W] or [1, H, W] or [H, W]

    if is_depth:
        x = visualize_depth(x)
        plt.imshow(x)
        plt.title(title)
        plt.colorbar()
        plt.savefig(path, bbox_inches='tight', dpi=200)
        plt.clf()
    else:
        plt.imshow(x)
        plt.title(title)
        plt.savefig(path, bbox_inches='tight', dpi=200)
        plt.clf()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'


def predefined_val_ts(img_id):

    aoi_id = img_id[:7]

    if aoi_id == "JAX_068":
        d = {"JAX_068_022_RGB": 8,
             "JAX_068_002_RGB": 8,
             "JAX_068_012_RGB": 1,
             "JAX_068_013_RGB": 1,
             "JAX_068_011_RGB": 1} #3
    elif aoi_id == "JAX_004":
        d = {"JAX_004_022_RGB": 0,
             "JAX_004_015_RGB": 0,
             "JAX_004_014_RGB": 0,
             "JAX_004_009_RGB": 5}
    elif aoi_id == "JAX_214":
        d = {"JAX_214_020_RGB": 0,
             "JAX_214_007_RGB": 8,
             "JAX_214_006_RGB": 8,
             "JAX_214_001_RGB": 18,
             "JAX_214_008_RGB": 2,
             "JAX_214_011_RGB": 17}
    elif aoi_id == "JAX_260":
        d = {"JAX_260_015_RGB": 0,
             "JAX_260_006_RGB": 3,
             "JAX_260_004_RGB": 10,
             "JAX_260_011_RGB": 10}
    else:
        return None
    return d[img_id]

def get_parameters(models):
    """
    Get all model parameters recursively
    models can be a list, a dictionary or a single pytorch model
    """
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:
        # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters



def save_as_plot(stat, epochs, title="", xlabel="", ylabel="", vecto=False, path=None, color='y', is_time=False):

    if path is None:
        print("**************No path for metrics provided**************")
    else:
        if is_time:
            plt.plot(stat, color, label=title) # pas une liste de liste
        else:
            plt.plot(_flatten(stat), color, label=title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

        if vecto:
            plt.savefig(path+'/'+title+" epoch= "+epochs+'.pdf', bbox_inches='tight') # vectorizé
        else:

            plt.savefig(path+'/'+title+'.png', bbox_inches='tight') # rastérizé ou non maybe avec ,rasterized=False
        plt.clf()

def plot_train_stats(stats, epochs, path, only_mse=False):

    save_as_plot(stats["loss"], epochs=epochs, title="Loss train",xlabel="Itérations", ylabel="Loss", path=path, color=('#008000')) # green
    save_as_plot(stats["ssim"], epochs=epochs, title="SSIM train",xlabel="Itérations", ylabel="SSIM", path=path, color=('#00FFCD')) # cyan
    save_as_plot(stats["psnr"], epochs=epochs, title="PSNR train",xlabel="Itérations", ylabel="PSNR (dB)", path=path, color=('#5E00FF')) # fuchsia
    save_as_plot(stats["mse"], epochs=epochs, title="MSE train",xlabel="Itérations", ylabel="MSE", path=path, color=('#CC00CC')) # vert foncé

    if not only_mse:
        save_as_plot(stats["sc"], epochs=epochs, title="Correction_solaire_train",xlabel="Epochs", ylabel="SC", path=path, color=('#FFD21A')) # yellow
    save_as_plot(stats["time"], epochs=epochs, title="Temps d'entrainement par epoch",xlabel="Epochs", ylabel="Time (sec)", path=path, color=('#99004C'), is_time=True) # purple


def _flatten(l):
    return [item.cpu() for sublist in l for item in sublist]