import math
import numpy as np
import torch
import torch.nn as nn
import raymarching
from .utils import custom_meshgrid


class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 grid_size=128,
                 ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = grid_size
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.

        self.numite = 0
        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            # print("\n self.cascade ", self.cascade)
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0
    
    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0
    
    

    def run_cuda_sat(self, rays_o, rays_d, sun_d=None, ts=None, nears=None, fars=None, dt_gamma=0, bg_color=1, perturb=False, force_all_rays=True, max_steps=256, T_thresh=1e-3 , **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        sun_d = sun_d.contiguous().view(-1, 3)

        if ts is None:
            N = rays_o.shape[0] # num rays
            M = N * max_steps # init max points number in total
            torch.zeros(M, 4, dtype=rays_o.dtype, device=rays_o.device)

        ts = ts.contiguous().view(-1, 4) # 4 == t_embbeding_tau


        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculated near far while rays generation
        nears, fars =  nears.squeeze(), fars.squeeze()

        results = {}
        align=-1 # old val 128

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1
            xyzs, dirs, deltas, sun_dirs, rays, all_ts = raymarching.march_rays_train(rays_o, rays_d, sun_d, ts, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, align, force_all_rays, dt_gamma, max_steps) 
            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

            sigmas, rgbs, sun_v, sky_color, beta = self(xyzs, dirs, sun_dirs, all_ts)
          

            irradiance = torch.add(sun_v,torch.mul((1 - sun_v) , sky_color))
            rgbs = torch.mul(rgbs, irradiance)
            sigmas = torch.mul(self.density_scale, sigmas)

            # we recover the maximum number of points along a ray to initiate composite_rays_train matrices, 
            # as there are not the same number of points on all rays
            max_point=int(torch.max(rays[:,2])) 

            weights_sum, depth, image, transparency, opacity, weights, shade, uncert = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, sun_v, beta, T_thresh, max_point)

            image = image.view(*prefix, 3)
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            depth = depth.view(*prefix)
                               
            results['weights_sum'] = weights_sum
            results['transparency'] = transparency
            results['opacity'] = opacity
            results['weights'] = weights
            results['shade'] = shade # s-nerf
            results['uncert'] = uncert # beta de xi tj
            

        else:
           
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            # dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            uncert_scene = torch.zeros(N, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            while step < max_steps:
                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas, sun_dirs, all_ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, sun_d, ts, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, align, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs, sun_v, sky_color, beta = self(xyzs, dirs, sun_dirs, all_ts)
                irradiance = torch.add(sun_v,torch.mul((1 - sun_v) , sky_color))
                rgbs = torch.mul(rgbs, irradiance)

                # sigmas = self.density_scale * sigmas
                sigmas = torch.mul(self.density_scale, sigmas)

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, beta, deltas, weights_sum, depth, image, uncert_scene, T_thresh)
                
                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step
            
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
            uncert_scene = uncert_scene.view(*prefix)

            results['uncert_scene'] = uncert_scene
        


        results['depth'] = depth
        results['image'] = image.squeeze(0)
        results['sun_v'] = sun_v

        
        return results


    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.
        S = self.grid_size # before, grid_size was harcoded at 128
        if not self.cuda_ray:
            return 
        
        ### update density grid

        tmp_grid = - torch.ones_like(self.density_grid)
        
        # full update.
        if self.iter_density < 16:
        #if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:
                        
                        # construct points
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long() # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query density
                            
                            sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                            sigmas *= self.density_scale
                            # assign 
                            tmp_grid[cas, indices] = sigmas

        # partial update (half the computation)
        else:
            N = self.grid_size ** 3 // 4 # H * H * H / 4
            for cas in range(self.cascade):
                # random sample some positions
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_bitfield.device) # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long() # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1) # [Nz]
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device)
                occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # query density
                sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                sigmas *= self.density_scale
                # assign 
                tmp_grid[cas, indices] = sigmas


        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).detach() # -1 regions are viewed as 0 density.
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().detach() / total_step)
        self.local_step = 0

       
    def render(self, rays_o, rays_d, rays_sun=None, ts=None, nears=None, fars=None, staged=False, max_ray_batch=4096, **kwargs):
        
        _run = self.run_cuda_sat
        results = _run(rays_o, rays_d, rays_sun, ts, nears, fars, **kwargs)

        return results
