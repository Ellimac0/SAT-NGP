import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    # '-O3', '-std=c++14',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
    #     '-I/'+os.getcwd()+'/ngp_venv/lib/python3.8/site-packages/nvidia/cusparse/include',
    #  '-I/'+os.getcwd()+'/ngp_venv/lib/python3.8/site-packages/nvidia/cublas/include',
    # '-I/'+os.getcwd()+'/ngp_venv/lib/python3.8/site-packages/nvidia/cusolver/include',
    # '-I/'+os.getcwd()+'/ngp_venv/lib/python3.8/site-packages/nvidia/curand/include',
    # '-I/'+os.getcwd()+'/ngp_venv/lib/python3.8/site-packages/nvidia/cudnn/include',
    #  '-I/'+os.getcwd()+'/ngp_venv/lib/python3.8/site-packages/nvidia/nccl/include',
    #  '-I/'+os.getcwd()+'/ngp_venv/lib/python3.8/site-packages/nvidia/nvtx/include',
    #  '-I/'+os.getcwd()+'/ngp_venv/lib/python3.8/site-packages/nvidia/cufft/include',
    # '-I/'+os.getcwd()+'/ngp_venv/lib/python3.8/site-packages/nvidia/cuda_runtime/include',
    # '-I/'+os.getcwd()+'/ngp_venv/lib/python3.8/site-packages/nvidia/cuda_nvrtc/include',
    #  '-I/'+os.getcwd()+'/ngp_venv/lib/python3.8/site-packages/nvidia/cuda_cupti/include'
]


if os.name == "posix":
    c_flags = ['-O3', '-std=c++14']
elif os.name == "nt":
    c_flags = ['/O2', '/std:c++17']

    # find cl.exe
    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

_backend = load(name='_sh_encoder',
                extra_cflags=c_flags,
                extra_cuda_cflags=nvcc_flags,
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'shencoder.cu',
                    'bindings.cpp',
                ]],
                )

__all__ = ['_backend']