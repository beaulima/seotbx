import logging
import seotbx
logger = logging.getLogger("seotbx.snappy.apps.gpt")

def parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="execute a generic GPT processing")
    ap.add_argument('args', nargs=argparse.REMAINDER)


def application_func(args):
    seotbx.snappy.utils.check_esa_snap_installation()
    command_args = args.args
    seotbx.snappy.core.run_gpt_base(command_args)


def registering_application():
    from seotbx.apps import registering_application as ra
    ra("gpt", parser_func, application_func)