import logging
import seotbx
import seotbx.snappy_api.sentinel2.s2atcor
logger = logging.getLogger("seotbx.snappy_api.apps")


def s2atcor_parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="atmospheric correction on Sentinel2 with bands extraction in tif")
    ap.add_argument('args', nargs=argparse.REMAINDER)


def s2atcor_application_func(args):
    seotbx.snappy_api.utils.check_esa_snap_installation()
    seotbx.snappy_api.sentinel2.s2atcor.s2atcor(args)


def gpt_parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="execute a generic GPT processing")
    ap.add_argument('args', nargs=argparse.REMAINDER)


def gpt_application_func(args):
    seotbx.snappy_api.utils.check_esa_snap_installation()
    command_args = args.args
    seotbx.snappy_api.core.run_gpt_base(command_args)


def registering_application():
    from seotbx.apps import registering_application as ra
    ra("gpt", gpt_parser_func, gpt_application_func)
    ra("s2atcor", s2atcor_parser_func, s2atcor_application_func)
    ra("s1preprocess", seotbx.snappy_api.sentinel1.s1preprocess.s1process_parser_func,
       seotbx.snappy_api.sentinel1.s1preprocess.s1process_application_func)




