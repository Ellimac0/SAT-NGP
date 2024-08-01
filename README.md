<div align="center">

# SAT-NGP


<p align="center">
  
  [<a href="https://arxiv.org/pdf/2403.18711"><strong>ArXiv</strong></a>]  
</p>


</div>


Official implementation of **SAT-NGP**, as presented in our paper: \
\
**SAT-NGP : Unleashing Neural Graphics Primitives for Fast Relightable Transient-Free 3D reconstruction from Satellite Imagery (IGARSS 2024)** \
*[Camille Billouard](https://fr.linkedin.com/in/camille-b-3v1415926) <sup>1</sup>, [Dawa Derksen](https://www.semanticscholar.org/author/Dawa-Derksen/8090472) <sup>1</sup>, 
[Emmanuelle Sarrazin](https://ieeexplore.ieee.org/author/37086503757) <sup>1</sup>
and [Bruno Vallet](https://www.umr-lastig.fr/bruno-vallet/) <sup>2</sup>* \
<sup>1</sup>CNES, <sup>2</sup>Univ Gustave Eiffel, ENSG, IGN, LASTIG, F-94160 



## Environment Setup


#### What's new here? ( 2024. 08. 01) 

## Updates

### Recent Fixes and Improvements

#### 1. Fixed `ZeroDivisionError` in `dsmr.py`

A `ZeroDivisionError` was occurring in the `mean_std` function within `dsmr.py` due to a division by zero when the `count` value was zero. This issue has been resolved by adding checks and adjustments for zero values. The relevant section of `dsmr.py` has been updated as follows:

- Added a check to replace `count` with a small value if it is zero.
- Added a small value to `sigu` and `sigv` if they are zero to avoid division by zero.

#### 2. Updated `torch.cuda.amp.autocast` Usage

The `torch.cuda.amp.autocast` function has been deprecated and replaced with `torch.amp.autocast('cuda', args...)`. The relevant sections in `trainer.py` have been updated accordingly to use the new function signature:

- Replaced `torch.cuda.amp.autocast` with `torch.amp.autocast('cuda', args...)`.

#### 3. Updated to CUDA Version 12.2

The code has been updated to support CUDA version 12.2. Ensure you have the appropriate CUDA version installed in your environment to utilize these changes effectively.


### Tested configurations :

| CPU/GPU         |    runs     |     
|-----------------|---------|
| AMD EPYC Milan 7713 / NVIDIA A100   | ✅      | 
| Intel Xeon E5-2698  / NVIDIA V100   | ❌      |  
| AMD EPYC 7742 / NVIDIA A100         | ✅      | 
| Intel Core i512400F / RTX 4060Ti    | ✅      | 
| AMD EPYC Milan 7713 / NVIDIA A40    | ✅      | 

### Create conda env
```bash
conda create -p satngp -y python=3.8
conda activate MY_PATH/satngp
python -m pip install --upgrade pip
python -m pip install setuptools==69.5.1 
```
### Install lib

```bash
conda install anaconda::libtiff -y
conda install libnvjpeg-dev -c nvidia -y
conda install -c conda-forge ncurses -y
conda install gdal==3.4.1 libgdal -y
conda install -c anaconda git -y
```

### Dependencies 
Install PyTorch with CUDA (this repo has been tested with CUDA 11.7) : 

```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit -y
```

### Environment variables 
```bash
cd satngp
ln -s lib lib64 # we want to avoid problems when compiling some packages
export LD_LIBRARY_PATH="$PWD/lib64:$LD_LIBRARY_PATH"
export VENV_LIB_PATH="$PWD/lib64/python3.8/site-packages/"
LDFLAGS="-L$PWD/lib"
export CUDA_PATH=$PWD
cd ..
```

### Getting the repo

```bash
git clone https://github.com/Ellimac0/SAT-NGP.git
cd SAT-NGP/
# https://github.com/rusty1s/pytorch_scatter
```

### Install and compile Pytorch Scatter
```bash
mkdir dep_ext
cd dep_ext
git clone --branch pytorch_1_11 https://github.com/rusty1s/pytorch_scatter/ 
cd pytorch_scatter
pip install . -vvv # -vvv is verbose for debugging during installation
cd ../..
```

### Install requirements

```bash
cd SAT-NGP
pip install -r requirements.txt
```

## Dataset

The data came from the [DFC2019 dataset](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019) and the link below from [SAT-NeRF](https://github.com/centreborelli/satnerf/releases/download/EarthVision2022/dataset.zip)

```bash
mkdir data
cd data
wget https://github.com/centreborelli/satnerf/releases/download/EarthVision2022/dataset.zip
unzip dataset.zip -d dataset
```

## Train
```bash
# in SAT-NGP
# may take a few minutes the first time, as the backend is compiled at .cache/torch_extensions/py38_cu117/ 
bash scripts/run_sat_ngp.sh data/dataset JAX_XXX 60000 1024
```


## Citation

Accepted to IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2024.

```
@misc{billouard2024satngp,
      title={SAT-NGP : Unleashing Neural Graphics Primitives for Fast Relightable Transient-Free 3D reconstruction from Satellite Imagery}, 
      author={Camille Billouard and Dawa Derksen and Emmanuelle Sarrazin and Bruno Vallet},
      year={2024},
      eprint={2403.18711},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgements
This work was performed using HPC resources from CNES Computing Center (DOI 10.24400/263303/CNES C3). The authors would like to thank the Johns Hopkins University Applied Physics Laboratory and IARPA for providing the data used in this study, and the IEEE GRSS Image Analysis and Data Fusion Technical Committee for organizing the Data Fusion Contest.
A portion of this work was build on top of : 


* Credits to [Jiaxiang Tang](https://github.com/ashawkey/torch-ngp/tree/main) for excellent work :

    ```
    @misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
    }
    ```
the contributors of : 

* [Pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) 

and authors of : 
* [SatNerf](https://github.com/centreborelli/satnerf) 



----


