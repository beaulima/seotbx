import logging
import os
import seotbx
import seotbx.polsarproc.definitions as defs
import seotbx.dl.apps.train

logger = logging.getLogger("seotbx.polsarproc.dll.apps")


def registering_application():

    from seotbx.apps import registering_application as ra
    ra("train",
       seotbx.dl.apps.train.train_parser_func,
       seotbx.dl.apps.train.train_application_func)
