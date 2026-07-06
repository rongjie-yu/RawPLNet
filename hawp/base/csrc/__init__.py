from torch.utils.cpp_extension import load
import os.path as osp
import os

__this__ = osp.dirname(__file__)
_repo_root = osp.abspath(osp.join(__this__, "..", "..", ".."))
_default_build_dir = osp.join(_repo_root, "outputs", "torch_extensions")


def _extension_build_dir():
    return os.environ.get(
        "HAWP_TORCH_EXTENSIONS_DIR",
        os.environ.get("TORCH_EXTENSIONS_DIR", _default_build_dir),
    )


def _load_extension():
    build_dir = _extension_build_dir()
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="_C",
        sources=[
            osp.join(__this__, "binding.cpp"),
            osp.join(__this__, "linesegment.cu"),
        ],
        build_directory=build_dir,
    )


try:
    _C = _load_extension()
    _C_LOAD_ERROR = None
except Exception as exc:
    _C = None
    _C_LOAD_ERROR = exc


def require_C():
    if _C_LOAD_ERROR is None:
        return _C
    raise RuntimeError(
        "HAWP C/CUDA extension is not available. Build prerequisites are required "
        "for line training because HAFM encoding calls `_C.encodels`. Ensure CUDA, "
        "nvcc, a compatible C++ compiler, and ninja are installed, then verify with "
        "`python -c \"from hawp.base.csrc import require_C; print(require_C().encodels)\"`. "
        f"Original extension load error: {_C_LOAD_ERROR}"
    ) from _C_LOAD_ERROR

__all__ = ["_C", "require_C"]

#_C = load(name='base._C', sources=['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'])
