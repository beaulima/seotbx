import logging
import torch
import os
import glob
import seotbx
from  seotbx.utils import str2bool
logger = logging.getLogger(__name__)
import inspect
import pathlib
import importlib
import importlib.util
import functools

def import_class(fullname):
    # type: (str) -> Type
    """General-purpose runtime class importer.
    Supported syntax:
        1. ``module.package.Class`` will import the fully qualified ``Class`` located
           in ``package`` from the *installed* ``module``
        2. ``/some/path/mod.pkg.Cls`` will import ``Cls`` as fully qualified ``mod.pkg.Cls`` from
           ``/some/path`` directory
    Args:
        fullname: the fully qualified class name to be imported.
    Returns:
        The imported class.
    """
    if inspect.isclass(fullname):
        return fullname  # useful shortcut for hacky configs
    assert isinstance(fullname, str), "should specify class by its (fully qualified) name"
    fullname = pathlib.Path(fullname).as_posix()
    if "/" in fullname:
        mod_path, mod_cls_name = fullname.rsplit("/", 1)
        pkg_name = mod_cls_name.rsplit(".", 1)[0]
        pkg_file = os.path.join(mod_path, pkg_name.replace(".", "/")) + ".py"
        spec = importlib.util.spec_from_file_location(mod_cls_name, pkg_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class_name = mod_cls_name.rsplit('.', 1)[-1]
    else:

        module_name, class_name = fullname.rsplit('.', 1)
        module = importlib.import_module(module_name)
    return getattr(module, class_name)


def import_function(func,
                    params=None
                    ):
    """General-purpose runtime function importer, with support for parameter binding.
    Args:
        func: the fully qualified function name to be imported, or a dictionary with
            two members (a ``type`` and optional ``params``), or a list of any of these.
        params: optional params dictionary to bind to the function call via functools.
            If a dictionary of parameters is also provided in ``func``, both will be merged.
    Returns:
        The imported function, with optionally bound parameters.
    """
    assert isinstance(func, (str, dict, list)) or callable(func), "invalid target function type"
    assert params is None or isinstance(params, dict), "invalid target function parameters"
    params = {} if params is None else params
    if isinstance(func, list):
        def multi_caller(funcs, *args, **kwargs):
            return [fn(*args, **kwargs) for fn in funcs]
        return functools.partial(multi_caller, [import_function(fn, params) for fn in func])
    if isinstance(func, dict):
        errmsg = "dynamic function import via dictionary must provide 'type' and 'params' members"
        fn_type = seotbx.utils.get_key(["type", "func", "function", "op", "operation", "name"], func, msg=errmsg)
        fn_params = seotbx.utils.get_key_def(["params", "param", "parameters", "kwargs"], func, None)
        fn_params = {} if fn_params is None else fn_params
        fn_params = {**params, **fn_params}
        return import_function(fn_type, params=fn_params)
    if isinstance(func, str):
        func = import_class(func)
    assert callable(func), f"unsupported function type ({type(func)})"
    if params:
        return functools.partial(func, **params)
    return func

def test_cuda_device_availability(device_idx):
    # type: (int) -> bool
    """Tests the availability of a single cuda device and returns its status."""
    # noinspection PyBroadException
    try:
        torch.cuda.set_device(device_idx)
        test_val = torch.cuda.FloatTensor([1])
        return test_val.cpu().item() == 1.0
    except Exception:
        return False


def get_available_cuda_devices(attempts_per_device=5):
    # type: (Optional[int]) -> List[int]
    """
    Tests all visible cuda devices and returns a list of available ones.
    Returns:
        List of available cuda device IDs (integers). An empty list means no
        cuda device is available, and the app should fallback to cpu.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return []
    devices_available = [False] * torch.cuda.device_count()
    attempt_broadcast = False
    for attempt in range(attempts_per_device):
        for device_id in range(torch.cuda.device_count()):
            if not devices_available[device_id]:
                if not attempt_broadcast:
                    logger.debug("testing availability of cuda device #%d (%s)" % (
                        device_id, torch.cuda.get_device_name(device_id)
                    ))
                devices_available[device_id] = test_cuda_device_availability(device_id)
        attempt_broadcast = True
    return [device_id for device_id, available in enumerate(devices_available) if available]


def setup_cudnn(config):
    """Parses the provided config for CUDNN flags and sets up PyTorch accordingly."""
    if "cudnn" in config and isinstance(config["cudnn"], dict):
        config = config["cudnn"]
        if "benchmark" in config:
            cudnn_benchmark_flag = str2bool(config["benchmark"])
            logger.debug("cudnn benchmark mode = %s" % str(cudnn_benchmark_flag))
            torch.backends.cudnn.benchmark = cudnn_benchmark_flag
        if "deterministic" in config:
            cudnn_deterministic_flag = str2bool(config["deterministic"])
            logger.debug("cudnn deterministic mode = %s" % str(cudnn_deterministic_flag))
            torch.backends.cudnn.deterministic = cudnn_deterministic_flag
    else:
        if "cudnn_benchmark" in config:
            cudnn_benchmark_flag = str2bool(config["cudnn_benchmark"])
            logger.debug("cudnn benchmark mode = %s" % str(cudnn_benchmark_flag))
            torch.backends.cudnn.benchmark = cudnn_benchmark_flag
        if "cudnn_deterministic" in config:
            cudnn_deterministic_flag = str2bool(config["cudnn_deterministic"])
            logger.debug("cudnn deterministic mode = %s" % str(cudnn_deterministic_flag))
            torch.backends.cudnn.deterministic = cudnn_deterministic_flag






