import logging

APPLICATIONS_TABLE = {}

def registering_application(application_name, parser_func, application_func):
    APPLICATIONS_TABLE[application_name] = {"parser_func":parser_func, "application_func":application_func}

def get_applications_table():
    return APPLICATIONS_TABLE



logger = logging.getLogger("seotbx.apps")
