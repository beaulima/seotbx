"""Top-level package for the 'seotbx' framework.
Running ``import seotbx`` will recursively import all important subpackages and modules.
"""

import logging
import seotbx.cli
import seotbx.utils
import seotbx.apps
import seotbx.dl
import seotbx.utils
import seotbx.polsarproc
import seotbx.snappy
import seotbx.geoserver
import seotbx.sentinelsat_api
import seotbx.slurm

#applications
seotbx.polsarproc.apps.registering_application()
seotbx.snappy.apps.gpt.registering_application()
seotbx.sentinelsat_api.apps.download_sentinel1.registering_application()
seotbx.snappy.apps.s2atcor.registering_application()

logger = logging.getLogger("seotbx")

__url__ = "https://github.com/beaulima/seotbx"
__version__ = "0.1.0"