#! /bin/bash


aoi_id=$2 # "JAX_214"
gpu_id=0
downsample_factor=1 # 1
training_iters=$3 # 30000 for batch==20248 
errs="${aoi_id}_errors.txt"

data_dir=$1
DFC2019_dir="${data_dir}/DFC2019"
root_dir="${data_dir}/root_dir/crops_rpcs_ba_v2/${aoi_id}"
cache_dir="${data_dir}/cache_dir/crops_rpcs_ba_v2/${aoi_id}_ds${downsample_factor}"

exec_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
script_dir=$( cd -- "$( dirname -- "${script_dir}" )" &> /dev/null && pwd )

img_dir="${DFC2019_dir}/Track3-RGB-crops/${aoi_id}"


# # # -------------------------------------------------------------------------------------------------------
# # -------------------------------------------------------------------------------------------------------
# # TRAIN
# # # # -------------------------------------------------------------------------------------------------------
nvidia-smi 


ckpt="scratch"
skips=-1
num_layers=2 # 2
hidden_dim=64 # 64
max_steps=256 # 256
encoding="hashgrid"
encoding_dir="sphere_harmonics"
loss="sat"

optim="radam" # radam adam
lr=0.01 # default==0.01
beta_1=0.9 # default==0.9
beta_2=0.99 #  default==0.99
weight_decay=0.000001 #  default==1e-6


init_weight="O"  # U=SIREN O=Robust
nonlinearity="leaky_relu" # leaky_relu relu
gain_nonlinearity="leaky_relu"  # relu leaky_relu tanh
mode="fan_in" # fan_in fan_out



first_beta_epoch=1 # add beta loss after $ epoch
lambda_sc=0.05 # 0.05

grid_size=128 # 128
num_levels=8 # 8
desired_resolution=0 # 4096 but 8192 better 16384 or 0
level_dim=2 # 4
base_resolution=16 # 16
log2_hashmap_size=19 # 19

color_space="srgb" # linear, srgb

bound=1 # 1
density_thresh=20 # 20 tests
density_scale=1 # 1
degree=3 # 3
update_extra_interval=16
dt_gamma=0

batch_size=$4 # 

model="ngp-sat-nerf"
exp_name="${aoi_id}_ds${downsample_factor}_${model}_iter_${training_iters}_batch_${batch_size}"
out_dir="${script_dir}/outputs/train/${exp_name}"
logs_dir="${out_dir}/logs"
ckpts_dir="${out_dir}/ckpts"
errs_dir="${out_dir}/errs"
mkdir -p $errs_dir
mkdir -p $logs_dir
gt_dir="${DFC2019_dir}/Track3-Truth"


custom_args="--exp_name ${exp_name} --model $model --img_downscale ${downsample_factor} --max_train_steps ${training_iters} --fc_units 256"
errs="${errs_dir}/${exp_name}_errors.txt"
echo -n "" > $errs

# Create temp directory
mkdir -p temp/
start=$(date +%s)
python3 ${script_dir}/main_nerf.py data/sat --init_weight ${init_weight} --nonlinearity ${nonlinearity} --mode ${mode} --gain_nonlinearity ${gain_nonlinearity} --degree ${degree} --lr ${lr} --beta_1 ${beta_1} --beta_2 ${beta_2} --weight_decay ${weight_decay} --color_space ${color_space} --batch_size ${batch_size} --first_beta_epoch ${first_beta_epoch} --update_extra_interval ${update_extra_interval} --ckpt $ckpt --grid_size ${grid_size} --dt_gamma ${dt_gamma} --num_levels ${num_levels} --density_scale ${density_scale} --density_thresh ${density_thresh} --bound $bound --desired_resolution ${desired_resolution}  --level_dim ${level_dim} --base_resolution ${base_resolution} --log2_hashmap_size ${log2_hashmap_size} --lambda_sc ${lambda_sc} --optim $optim --loss $loss --encoding $encoding --encoding_dir ${encoding_dir} --num_layers ${num_layers} --hidden_dim ${hidden_dim} --skips $skips --workspace ${out_dir} --gt_dir ${gt_dir} --max_steps ${max_steps} --iters ${training_iters} --root_dir ${root_dir} --img_dir ${img_dir} --cache_dir ${cache_dir} --downscale ${downsample_factor} --fp16 --cuda_ray
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
