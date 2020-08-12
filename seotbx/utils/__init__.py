import seotbx.utils.logger
import seotbx.utils.const
import os
from datetime import datetime as dt

DEFAULT_DT_STR_FORMAT = "%d-%b-%Y-(%H:%M:%S.%f)"
SHORT_DT_STR_FORMAT = "%d-%b-%Y"

def get_now():
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

