import logging
import os
import seotbx
import seotbx.polsarproc.definitions as defs
import seotbx.polsarproc.apps.create_polsar_signatures
import numpy as np

logger = logging.getLogger("seotbx.polsarproc.apps")

def registering_application():
    from seotbx.apps import registering_application as ra
    ra("polsarsigs",
       seotbx.polsarproc.apps.create_polsar_signatures.polsarsigs_parser_func,
       seotbx.polsarproc.apps.create_polsar_signatures.polsarsigs_application_func)



