import logging
import os
import seotbx
import seotbx.polsarproc.definitions as defs
import seotbx.polsarproc.apps.create_polsar_signatures
import seotbx.polsarproc.apps.create_polsar_samples
import numpy as np

logger = logging.getLogger("seotbx.polsarproc.apps")

def registering_application():
    from seotbx.apps import registering_application as ra
    ra("polsarsigs",
       seotbx.polsarproc.apps.create_polsar_signatures.polsarsigs_parser_func,
       seotbx.polsarproc.apps.create_polsar_signatures.polsarsigs_application_func)
    ra("polsarsamples",
       seotbx.polsarproc.apps.create_polsar_samples.polsarsamples_parser_func,
       seotbx.polsarproc.apps.create_polsar_samples.polsarsamples_application_func)



