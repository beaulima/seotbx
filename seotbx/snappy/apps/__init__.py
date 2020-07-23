import logging
import seotbx
import seotbx.snappy.sentinel2.s2atcor
logger = logging.getLogger("seotbx.snappy.apps")


def s2atcor_parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="atmospheric correction on Sentinel2 with bands extraction in tif")
    ap.add_argument('args', nargs=argparse.REMAINDER)


def s2atcor_application_func(args):
    seotbx.snappy.utils.check_esa_snap_installation()
    seotbx.snappy.sentinel2.s2atcor.s2atcor(args)

def gpt_parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="execute a generic GPT processing")
    ap.add_argument('args', nargs=argparse.REMAINDER)

def gpt_application_func(args):
    seotbx.snappy.utils.check_esa_snap_installation()
    command_args = args.args
    seotbx.snappy.core.run_gpt_base(command_args)

def registering_application():
    from seotbx.apps import registering_application as ra
    ra("gpt", gpt_parser_func, gpt_application_func)
    ra("s2atcor", s2atcor_parser_func, s2atcor_application_func)




