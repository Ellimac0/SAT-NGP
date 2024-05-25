import os
import glob
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import rasterio
import rpcm

import affine
from .sat_utils import get_file_id, read_dict_from_json, write_dict_to_json, rpc_scaling_params, rescale_rpc, utm_from_latlon




@torch.cuda.amp.autocast(enabled=False)
def get_sat_rays(cols, rows, rpc, min_alt, max_alt):
    """
            Draw a set of rays from a satellite image
            Each ray is defined by an origin 3d point + a direction vector
            First the bounds of each ray are found by localizing each pixel at min and max altitude
            Then the corresponding direction vector is found by the difference between such bounds
            Args:
                cols: 1d array with image column coordinates
                rows: 1d array with image row coordinates
                rpc: RPC model with the localization function associated to the satellite image
                min_alt: float, the minimum altitude observed in the image
                max_alt: float, the maximum altitude observed in the image
            Returns:
                rays: (h*w, 8) tensor of floats encoding h*w rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
            """
    max_alt*=1.2
    min_alts = float(min_alt) * np.ones(cols.shape)
    max_alts = float(max_alt) * np.ones(cols.shape)

    
    lons, lats = rpc.localization(cols, rows, max_alts)
    easts, norths, n, l = utm_from_latlon(lats, lons)
    xyz_near = np.vstack([easts, norths, max_alts]).T


    lons, lats = rpc.localization(cols, rows, min_alts)
    easts, norths, _, _ = utm_from_latlon(lats, lons)
    xyz_far = np.vstack([easts, norths, min_alts]).T

    rays_o = xyz_near 
    
    d = xyz_far - xyz_near

    rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]
    
    
    fars = np.linalg.norm(d, axis=1)
    nears = float(0) * np.ones(fars.shape)
    rays = torch.from_numpy(np.hstack([rays_o, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]]))

    return rays.double(), n, l

def load_tensor_from_rgb_geotiff(img_path, downscale_factor, is_training,
    imethod=T.InterpolationMode.BICUBIC
    ): 


    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0)) / 255.
    h, w = img.shape[:2]
    if downscale_factor > 1:
        w = int(w // downscale_factor)
        h = int(h // downscale_factor)
        img = np.transpose(img, (2, 0, 1))
        img = T.Resize(size=(h, w), interpolation=imethod, antialias=True)(torch.Tensor(img))
        img = np.transpose(img.numpy(), (1, 2, 0))

    # sat (h*w, 3)
    if is_training:
        img = T.ToTensor()(img)  # (3, h, w) 
        rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
    else:
        img = T.ToTensor()(img)  # (3, h, w) 
        rgbs = img.permute(1,2,0).unsqueeze(0) # (1, H, W, 3) 


    return rgbs, h, w

class SATNeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10, boundary_rayo_norma=(0, 0, 0, 0)):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU  .to(gpu device avant le train peut accel)
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        
        
        self.num_rays = self.opt.num_rays if self.training else -1

        self.min_alt, self.max_alt = -1000, 1000
        self.n, self.l = None, None
        self.ray_ox_min, self.ray_ox_max = boundary_rayo_norma[0], boundary_rayo_norma[1]
        self.ray_oy_min, self.ray_oy_max = boundary_rayo_norma[2], boundary_rayo_norma[3]

        self.batch_size = opt.batch_size
        
        self.first_time = 1 # we generate the dsm with ray direction diff 0, 0, -1
        
        # -----------
        self.json_dir = opt.root_dir
        self.img_dir = opt.img_dir
        self.cache_dir = opt.cache_dir
        self.gt_dir = opt.gt_dir

        assert os.path.exists(opt.root_dir), f"root_dir {opt.root_dir} does not exist"
        assert os.path.exists(opt.img_dir), f"img_dir {opt.img_dir} does not exist"

         # load scaling params
        if not os.path.exists(f"{self.json_dir}/scene_utm.loc"):
            self.init_scaling_params()
        
        d = read_dict_from_json(os.path.join(self.json_dir, "scene_utm.loc"))
        self.center = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])])
        self.range = torch.max(torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])]))

        self.all_rays, self.all_rgbs, self.all_ids = [], [], []
        
       
        # load dataset split
        if self.training:
            self.load_train_split()
        else:
            self.load_val_split()

    def load_train_split(self): 
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files if len(json_p) > 0]
        
        
        self.all_rays, self.all_rgbs, self.all_ids = self.load_data(self.json_files, verbose=True)
        
    def load_val_split(self):
        with open(os.path.join(self.json_dir, "test.txt"), "r") as f:
            json_files = f.read().split("\n")
            # print("test", json_files)
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        n_train_ims = len(json_files)
        self.all_ids = [i + n_train_ims for i, j in enumerate(self.json_files)]
        
    def init_scaling_params(self):
        print("Could not find a scene_utm.loc file in the root directory, creating one...")
        print("Warning: this can take some minutes")
        all_json = glob.glob("{}/*.json".format(self.json_dir))
        all_rays = []
        for json_p in all_json:
            d = read_dict_from_json(json_p)
            h, w = int(d["height"] // self.downscale), int(d["width"] // self.downscale)
            rpc = rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.downscale)
            min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
            cols, rows = np.meshgrid(np.arange(w), np.arange(h))
            rays, self.n, self.l = get_sat_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
            all_rays += [rays]
        all_rays = torch.cat(all_rays, 0)
        near_points = all_rays[:, :3]
        far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
        all_points = torch.cat([near_points, far_points], 0)

        d = {}
        d["X_scale"], d["X_offset"] = rpc_scaling_params(all_points[:, 0])
        d["Y_scale"], d["Y_offset"] = rpc_scaling_params(all_points[:, 1])
        d["Z_scale"], d["Z_offset"] = rpc_scaling_params(all_points[:, 2])
        write_dict_to_json(d, f"{self.json_dir}/scene_utm.loc")

    def load_data(self, json_files, for_dsm=False, verbose=True):
        """
        Load all relevant information from a set of json files
        Args:
            json_file: the path to the input json file
        Returns:
            all_rays: (N, 11) tensor of floats encoding all ray-related parameters corresponding to N rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
                      columns 8,9,10 correspond to the sun direction vectors
            all_rgbs: (N, 3) tensor of floats encoding all the rgb colors corresponding to N rays
        """
  
  
        all_rgbs, all_rays, all_sun_dirs, all_ids = [], [], [], []

        _get_n_l=True
        for t, json_p in enumerate(json_files):

            # read json, image path and id
            d = read_dict_from_json(json_p)
            img_p = os.path.join(self.img_dir, d["img"])
            img_id = get_file_id(d["img"])

            rgbs, h, w = load_tensor_from_rgb_geotiff(img_p, self.downscale, self.training, for_dsm)

            if _get_n_l: # we need n l for dsm generation 
                rpc = rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.downscale)
                self.min_alt, self.max_alt = float(d["min_alt"]), float(d["max_alt"])
                cols, rows = np.meshgrid(np.arange(w), np.arange(h))
                _, self.n, self.l = get_sat_rays(cols.flatten(), rows.flatten(), rpc, self.min_alt, self.max_alt)
                _get_n_l=False

            cache_path = "{}/{}.data".format(self.cache_dir, img_id)
            if self.cache_dir is not None and os.path.exists(cache_path):
                rays = torch.load(cache_path)
            else:
                
                rpc = rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.downscale)
                self.min_alt, self.max_alt = float(d["min_alt"]), float(d["max_alt"])
                cols, rows = np.meshgrid(np.arange(w), np.arange(h))
                
                rays, self.n, self.l = get_sat_rays(cols.flatten(), rows.flatten(), rpc, self.min_alt, self.max_alt)
                
                if self.cache_dir is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(rays, cache_path)
            
            rays = self.normalize_rays(rays)
            # get sun direction
            sun_dirs = self.get_sun_dirs(float(d["sun_elevation"]), float(d["sun_azimuth"]), rays.shape[0])

            all_ids += [t * torch.ones(rays.shape[0], 1)]
            all_rgbs += [rgbs]
            all_rays += [rays]
            all_sun_dirs += [sun_dirs]
            if verbose:
                print("Image {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))
            
        all_ids = torch.cat(all_ids, 0)
        all_sun_dirs = torch.cat(all_sun_dirs, 0)  # (len(json_files)*h*w, 3)
        all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        
        if not self.training and for_dsm:
            # direction nadir 
            all_rays[:, 3] = 0 
            all_rays[:, 4] = 0
            all_rays[:, 5] = -1 
            
        all_rays = torch.hstack([all_rays, all_sun_dirs])  # (len(json_files)*h*w, 11)
        all_rays = all_rays.type(torch.FloatTensor)
        
        all_rgbs = torch.cat(all_rgbs, 0)  # (len(json_files)*h*w, 3)
        all_rgbs = all_rgbs.type(torch.FloatTensor)
        
        return all_rays, all_rgbs, all_ids

    def normalize_rays(self, rays):
        rays[:, 0] -= self.center[0]
        rays[:, 1] -= self.center[1]
        rays[:, 2] -= self.center[2]
        rays[:, 0] /= self.range
        rays[:, 1] /= self.range
        rays[:, 2] /= self.range
        
        rays[:, 6] /= self.range
        rays[:, 7] /= self.range
        return rays
        
    def get_sun_dirs(self, sun_elevation_deg, sun_azimuth_deg, n_rays):
        """
        Get sun direction vectors
        Args:
            sun_elevation_deg: float, sun elevation in  degrees
            sun_azimuth_deg: float, sun azimuth in degrees
            n_rays: number of rays affected by the same sun direction
        Returns:
            sun_d: (n_rays, 3) 3-valued unit vector encoding the sun direction, repeated n_rays times
        """
        sun_el = np.radians(sun_elevation_deg)
        sun_az = np.radians(sun_azimuth_deg)
        sun_d = np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
        sun_dirs = torch.from_numpy(np.tile(sun_d, (n_rays, 1)))
        
        return sun_dirs

    def utm_from_UTM_nerf_prediction(self, data, depth):
     
        rays_o = data['rays_o'].squeeze(0).double() # [1, N, 3] 
        rays_d = data['rays_d'].squeeze(0).double() # [1, N, 3]

        depth = depth.double() 

        xyz_n = rays_o + rays_d * depth.view(-1, 1)
        xyz = xyz_n * self.range
        xyz[:, 0] += self.center[0]
        xyz[:, 1] += self.center[1]
        xyz[:, 2] += self.center[2]
        xyz = xyz.detach().cpu().numpy()
        return xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    def get_dsm_from_UTM_nerf_prediction(self, data, depth, dsm_path=None, roi_txt=None):
        """
        Compute a DSM from a NeRF depth prediction output
        Args:
            rays: (h*w, 11) tensor of input rays
            depth: (h*w, 1) tensor with nerf depth prediction
            dsm_path (optional): string, path to output DSM, in case you want to write it to disk
            roi_txt (optional): compute the DSM only within the bounds of the region of interest of the txt
        Returns:
            dsm: (h, w) numpy array with the output dsm
        """

        # get point cloud from nerf depth prediction
        easts, norths, alts = self.utm_from_UTM_nerf_prediction(data, depth)
        cloud = np.vstack([easts, norths, alts]).T

        
        # (optional) read region of interest, where lidar GT is available
        if roi_txt is not None:
            gt_roi_metadata = np.loadtxt(roi_txt)
            xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
            xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
            resolution = gt_roi_metadata[3]
            yoff += ysize * resolution  # weird but seems necessary ?
        else:
            resolution = 0.5 # 50 cm for dfc2019
            xmin, xmax = cloud[:, 0].min(), cloud[:, 0].max()
            ymin, ymax = cloud[:, 1].min(), cloud[:, 1].max()
            xoff = np.floor(xmin / resolution) * resolution
            xsize = int(1 + np.floor((xmax - xoff) / resolution))
            yoff = np.ceil(ymax / resolution) * resolution
            ysize = int(1 - np.floor((ymin - yoff) / resolution))

        from plyflatten import plyflatten
        from plyflatten.utils import rasterio_crs, crs_proj

        # run plyflatten
        dsm = plyflatten(cloud, xoff, yoff, resolution, xsize, ysize, radius=1, sigma=float("inf"))
        crs_proj = rasterio_crs(crs_proj("{}{}".format(self.n, self.l), crs_type="UTM"))
       
        # (optional) write dsm to disk
        if dsm_path is not None:
            os.makedirs(os.path.dirname(dsm_path), exist_ok=True)
            profile = {}
            profile["dtype"] = dsm.dtype
            profile["height"] = dsm.shape[0]
            profile["width"] = dsm.shape[1]
            profile["count"] = 1
            profile["driver"] = "GTiff"
            profile["nodata"] = float("nan")
            
            profile["crs"] = crs_proj
            profile["transform"] = affine.Affine(resolution, 0.0, xoff, 0.0, -resolution, yoff)
            with rasterio.open(dsm_path, "w", **profile) as f:
                f.write(dsm[:, :, 0], 1)
        else:
            print("(optional) write dsm to disk not DONE")

        return dsm

    def collate(self, index):
  
        # get rays
        if self.training:
            
            rays = self.all_rays[index]
            rays_o, rays_d, near, far, sun_d = rays[:, 0:3],  rays[:, 3:6], rays[:, 6:7], rays[:, 7:8], rays[:, 8:11]
            rgbs = self.all_rgbs[index]
            ts = self.all_ids[index]
            
            results = {
            'rgbs' : rgbs.to(self.device),
            'rays_o': rays_o.unsqueeze(0).to(self.device),
            'rays_d': rays_d.unsqueeze(0).to(self.device),
            'nears': near.to(self.device),
            'fars': far.to(self.device),
            'sun_d': sun_d.to(self.device),
            'ts': ts.long().to(self.device)
            }
            
                 
        else:
            json_file_idx = self.json_files[index[0]]
            
            d = read_dict_from_json(json_file_idx)
            if self.first_time%2==0: # first for NVS and second time for DSM
                for_dsm=True
                h, w = int(d["height"] // self.downscale), int(d["width"] // self.downscale)
       
            else:
                for_dsm=False
                h, w = int(d["height"] // self.downscale), int(d["width"] // self.downscale)
            
            rays, rgbs, _ = self.load_data([json_file_idx], for_dsm, verbose=True)
                
            self.first_time+=1
            
            rays_o, rays_d, near, far, sun_d = rays[:, 0:3],  rays[:, 3:6], rays[:, 6:7], rays[:, 7:8], rays[:, 8:11]
            ts = self.all_ids[index[0]] * torch.ones(rays.shape[0], 1)

            
            img_id = get_file_id(d["img"])
            
  
            results = {
            'rgbs' : rgbs.to(self.device),
            'rays_o': rays_o.unsqueeze(0).to(self.device),
            'rays_d': rays_d.unsqueeze(0).to(self.device),
            'nears': near.to(self.device),
            'fars': far.to(self.device),
            'sun_d': sun_d.to(self.device),
            'ts': ts.long().to(self.device),
            'src_id': img_id,
            'H' : h, 
            'W' : w
            }
            
           
       
        return results

    def dataloader(self):
        if self.training:
            size = self.all_rays.shape[0]
            print("dataloader train size =", size)
        else:
            size = len(self.json_files)
            print("dataloader test/val size*nbRay/image =", size)
       
        loader = DataLoader(list(range(size)), batch_size=self.batch_size, collate_fn=self.collate, 
                            shuffle=self.training, 
                            num_workers=0, pin_memory=False)
        loader._data = self # an ugly fix... we need poses in trainer.
        loader.has_gt = self.all_rgbs is not None
        return loader