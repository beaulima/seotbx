import logging
import seotbx
import seotbx.snappy.apps.s2atcor
logger = logging.getLogger(__name__)


def gpt_subparser(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="execute a generic GPT processing")
    ap.add_argument('args', nargs=argparse.REMAINDER)


def gpt_function(args):
    seotbx.snappy.utils.check_esa_snap_installation()
    command_args = args.args
    seotbx.snappy.core.run_gpt_base(command_args)


def registering_application():
    from seotbx.apps import registering_application as ra
    ra("gpt", gpt_subparser, gpt_function)

def run_gpt(args):
    import snaphelper
    #snaphelper.logger.debug("GPT Processing")
    command_args = args.args
