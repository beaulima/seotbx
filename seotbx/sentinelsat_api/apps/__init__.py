import logging
import seotbx

logger = logging.getLogger("seotbx.sentinelsat.apps")

def registering_application():
    from seotbx.apps import registering_application as ra
    ra("download_sentinel1",
       seotbx.sentinelsat_api.download_sentinel1.parser_func,
       seotbx.sentinelsat_api.download_sentinel1.application_func)
