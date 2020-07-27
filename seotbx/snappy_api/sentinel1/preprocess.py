import logging
import seotbx
import datetime
import os
import glob

logger = logging.getLogger("seotbx.sentinelsat.apps.preprocessing")


def parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="preprocessing sentinel1")
    ap.add_argument("-u", "--scihub_username", default=None, type=str, help="scihub user name")
    ap.add_argument("-p", "--scihub_passwd", default=None, type=str, help="scihub password name")

def application_func(args):
    return