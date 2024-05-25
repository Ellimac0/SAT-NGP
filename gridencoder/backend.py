import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    # '-O2', '-std=c++14',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
    # f'-I{nvidia_path}/cusparse/include',
    # f'-I{nvidia_path}/cublas/include',
    # f'-I{nvidia_path}/cusolver/include',
    # f'-I{nvidia_path}/curand/include',
    # f'-I{nvidia_path}/cudnn/include',
    # f'-I{nvidia_path}/nccl/include',
    # f'-I{nvidia_path}/nvtx/include',
    # f'-I{nvidia_path}/cufft/include',
    # f'-I{nvidia_path}/cuda_runtime/include',
    # f'-I{nvidia_path}/cuda_nvrtc/include',
    # f'-I{nvidia_path}/cuda_cupti/include'
]

if os.name == "posix":
    c_flags = ['-O2', '-std=c++14']
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

_backend = load(name='_grid_encoder',
                extra_cflags=c_flags,
                extra_cuda_cflags=nvcc_flags,
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'gridencoder.cu',
                    'bindings.cpp',
                ]],
                )

__all__ = ['_backend']
