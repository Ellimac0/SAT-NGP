#pragma once

#include <stdint.h>
#include <torch/torch.h>


void near_far_from_aabb(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t N, const float min_near, at::Tensor nears, at::Tensor fars);
void sph_from_ray(const at::Tensor rays_o, const at::Tensor rays_d, const float radius, const uint32_t N, at::Tensor coords);
void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices);
void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords);
void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield);

// modif SAT 
void composite_rays_train_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, 
                                 const at::Tensor rays,const at::Tensor sun_v,const at::Tensor beta,
                                 const uint32_t M, const uint32_t N, const float T_thresh, 
                                 at::Tensor weights_sum, at::Tensor depth, at::Tensor image, 
                                 at::Tensor shade, at::Tensor uncert, 
                                 at::Tensor transparency, at::Tensor opacity, at::Tensor weights, const uint32_t max_point);

void march_rays_train(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor sun_d, const at::Tensor ts, 
                    const at::Tensor grid, const float bound, const float dt_gamma, const uint32_t max_steps, 
                    const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M, const at::Tensor nears, const at::Tensor fars, 
                    at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor sun_dirs, at::Tensor all_ts, at::Tensor rays, at::Tensor counter, at::Tensor noises);


void march_rays(const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive, const at::Tensor rays_t, 
                const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor sun_d, const at::Tensor ts, const float bound, 
                const float dt_gamma, const uint32_t max_steps, const uint32_t C, const uint32_t H, 
                const at::Tensor grid, const at::Tensor near, const at::Tensor far, 
                at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor sun_dirs, at::Tensor all_ts, at::Tensor noises);


void composite_rays(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, 
                    at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor beta, 
                    at::Tensor deltas, at::Tensor weights, at::Tensor depth, at::Tensor image, at::Tensor uncert_scene);

void composite_rays_train_backward(const at::Tensor grad_depth, const at::Tensor grad_opacity, const at::Tensor grad_weights_sum, const at::Tensor grad_image, const at::Tensor sigmas, const at::Tensor rgbs, 
                                    const at::Tensor grad_uncert, const at::Tensor grad_shade, const at::Tensor beta, const at::Tensor sun_v, 
                                    const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum, const at::Tensor depth, const at::Tensor opacity,
                                    const at::Tensor image, const at::Tensor uncert, const at::Tensor shade, const uint32_t M, const uint32_t N, const float T_thresh,
                                     at::Tensor grad_sigmas, at::Tensor grad_rgbs, at::Tensor grad_sun_v, at::Tensor grad_betas);
