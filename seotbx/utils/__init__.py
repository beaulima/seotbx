import logging
import seotbx.utils.logger
import seotbx.utils.const
import os
from datetime import datetime as dt
import platform
import time
import seotbx
import sys


DEFAULT_DT_STR_FORMAT = "%d-%b-%Y-(%H:%M:%S.%f)"
SHORT_DT_STR_FORMAT = "%d-%b-%Y"

def get_now():
    """
    Returns the current datetime
    """
    return dt.now()


def get_current_time_string(format: str ="%d-%b-%Y-(%H:%M:%S.%f)") ->str:
    """
    Returns current date/time in string format
    params:
    format: Default format string
    """
    dtobj = dt.now()
    return dtobj.strftime(format)


def create_path_with_timestamp(dirpath: str, basename: str, ext: str,
                               format: str =SHORT_DT_STR_FORMAT,
                               dtobj = None  ) ->str:
    if dtobj is None:
        return os.path.join(dirpath, f"{basename}-{get_current_time_string(format)}.{ext}")
    else:
        return os.path.join(dirpath, f"{basename}-{dtobj.strftime(format)}.{ext}")


def get_key(key, config, msg=None, delete=False):
    # from thelper
    """Returns a value given a dictionary key, throwing if not available."""
    if isinstance(key, list):
        if len(key) <= 1:
            if msg is not None:
                raise AssertionError(msg)
            else:
                raise AssertionError("must provide at least two valid keys to test")
        for k in key:
            if k in config:
                val = config[k]
                if delete:
                    del config[k]
                return val
        if msg is not None:
            raise AssertionError(msg)
        else:
            raise AssertionError("config dictionary missing a field named as one of '%s'" % str(key))
    else:
        if key not in config:
            if msg is not None:
                raise AssertionError(msg)
            else:
                raise AssertionError("config dictionary missing '%s' field" % key)
        else:
            val = config[key]
            if delete:
                del config[key]
            return val


def get_key_def(key, config, default=None, msg=None, delete=False):
    # from thelper
    """Returns a value given a dictionary key, or the default value if it cannot be found."""
    if isinstance(key, list):
        if len(key) <= 1:
            if msg is not None:
                raise AssertionError(msg)
            else:
                raise AssertionError("must provide at least two valid keys to test")
        for k in key:
            if k in config:
                val = config[k]
                if delete:
                    del config[k]
                return val
        return default
    else:
        if key not in config:
            return default
        else:
            val = config[key]
            if delete:
                del config[key]
            return val


def str2size(input_str):
    """Returns a (WIDTH, HEIGHT) integer size tuple from a string formatted as 'WxH'."""
    if not isinstance(input_str, str):
        raise AssertionError("unexpected input type")
    display_size_str = input_str.split('x')
    if len(display_size_str) != 2:
        raise AssertionError("bad size string formatting")
    return tuple([max(int(substr), 1) for substr in display_size_str])


def str2bool(s):
    """Converts a string to a boolean.
    If the lower case version of the provided string matches any of 'true', '1', or
    'yes', then the function returns ``True``.
    """
    if isinstance(s, bool):
        return s
    if isinstance(s, (int, float)):
        return s != 0
    if isinstance(s, str):
        positive_flags = ["true", "1", "yes"]
        return s.lower() in positive_flags
    raise AssertionError("unrecognized input type")


def get_log_stamp():
    # from thelper
    """Returns a print-friendly and filename-friendly identification string containing platform and time."""
    return str(platform.node()) + "-" + time.strftime("%Y%m%d-%H%M%S")


def get_git_stamp():
    # from thelper
    """Returns a print-friendly SHA signature for the framework's underlying git repository (if found)."""
    try:
        import git
        try:
            repo = git.Repo(path=os.path.abspath(__file__), search_parent_directories=True)
            sha = repo.head.object.hexsha
            return str(sha)
        except (AttributeError, git.InvalidGitRepositoryError):
            return "unknown"
    except (ImportError, AttributeError):
        return "unknown"


def get_env_list():
    # from thelper
    """Returns a list of all packages installed in the current environment.
    If the required packages cannot be imported, the returned list will be empty. Note that some
    packages may not be properly detected by this approach, and it is pretty hacky, so use it with
    a grain of salt (i.e. logging is fine).
    """
    try:
        import pip
        # noinspection PyUnresolvedReferences
        pkgs = pip.get_installed_distributions()
        return sorted(["%s %s" % (pkg.key, pkg.version) for pkg in pkgs])
    except (ImportError, AttributeError):
        try:
            import pkg_resources as pkgr
            return sorted([str(pkg) for pkg in pkgr.working_set])
        except (ImportError, AttributeError):
            return []


def setup_cv2(config):
    """Parses the provided config for OpenCV flags and sets up its global state accordingly."""
    # https://github.com/pytorch/pytorch/issues/1355
    import cv2 as cv
    cv.setNumThreads(0)
    cv.ocl.setUseOpenCL(False)
    # todo: add more global opencv flags setups here


def setup_gdal(config):
    """Parses the provided config for GDAL flags and sets up its global state accordingly."""
    config = get_key_def("gdal", config, {})
    if "proj_search_path" in config:
        import osr
        osr.SetPROJSearchPath(config["proj_search_path"])


def setup_sys(config):
    """Parses the provided config for PYTHON sys paths and sets up its global state accordingly."""
    paths_to_add = []
    if "sys" in config:
        if isinstance(config["sys"], list):
            paths_to_add = config["sys"]
        elif isinstance(config["sys"], str):
            paths_to_add = [config["sys"]]
    for dir_path in paths_to_add:
        if os.path.isdir(dir_path):
            logger.debug(f"will append path to python's syspaths: {dir_path}")
            sys.path.append(dir_path)
        else:
            logger.warning(f"could not append to syspaths, invalid dir: {dir_path}")

def setup_plt(config):
    """Parses the provided config for matplotlib flags and sets up its global state accordingly."""
    import matplotlib.pyplot as plt
    config = get_key_def(["plt", "pyplot", "matplotlib"], config, {})
    if "backend" in config:
        import matplotlib
        matplotlib.use(get_key("backend", config))
    plt.interactive(get_key_def("interactive", config, False))


def setup_globals(config):
    """Parses the provided config for global flags and sets up the global state accordingly."""
    if "bypass_queries" in config and config["bypass_queries"]:
        global bypass_queries
        bypass_queries = True

    setup_sys(config)
    setup_plt(config)
    setup_cv2(config)
    setup_gdal(config)

    try:
        import torch
        seotbx.dl.utils.setup_cudnn(config)
    except ImportError:
        logger.info("pytorch not installed")
        pass

import importlib
import importlib.util
import pathlib
import inspect
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


def import_function(func,           # type: Union[Callable, AnyStr, List, Dict]
                    params=None     # type: Optional[thelper.typedefs.ConfigDict]
                    ):              # type: (...) -> FunctionType
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

def check_func_signature(func,      # type: FunctionType
                         params     # type: List[str]
                         ):         # type: (...) -> None
    """Checks whether the signature of a function matches the expected parameter list."""
    if func is None or not callable(func):
        raise AssertionError("invalid function object")
    if params is not None:
        if not isinstance(params, list) or not all([isinstance(p, str) for p in params]):
            raise AssertionError("unexpected param name list format")
        import inspect
        func_sig = inspect.signature(func)
        for p in params:
            if p not in func_sig.parameters:
                raise AssertionError("function missing parameter '%s'" % p)