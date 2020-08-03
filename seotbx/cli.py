import argparse
import logging
from typing import Any, Union

import seotbx

class Application:

    def __init__(self):
        self.apps_table = seotbx.apps.get_applications_table()
        self.argparser = self.make_argparser()

    def main(self, args=None, argparser=None):
        """Main entrypoint to use with console applications.
        """
        args = self.setup(args=args, argparser=argparser)
        if isinstance(args, int):
            return args  # CLI must exit immediately with provided error code

    def setup(self, args=None):
        # type: (Any) -> Union[int, argparse.Namespace]
        """Sets up the argument parser (if not already done externally) and parses the input CLI arguments.

        This function may return an error code (integer) if the program should exit immediately. Otherwise, it will return
        the parsed arguments to use in order to redirect the execution flow of the entrypoint.
        """

        args = self.argparser.parse_args(args=args)
        if args.version:
            print(seotbx.__version__)
            return 0
        if args.mode is None:
            self.argparser.print_help()
            return 1
        if args.silent and args.verbose > 0:
            raise AssertionError("contradicting verbose/silent arguments provided")
        log_level = logging.INFO if args.verbose < 1 else logging.DEBUG if args.verbose < 2 else logging.NOTSET
        seotbx.utils.logger.init_logger(seotbx, log_level, args.log, args.force_stdout)

        return args

    def make_argparser(self):
        """Creates the (default) argument parser to use for the main entrypoint.

        The argument parser will contain different "operating modes" that dictate the high-level behavior of the CLI. This
        function may be modified in branches of the framework to add project-specific features.
        """
        argparser = argparse.ArgumentParser(description='soetbx')
        argparser.add_argument("--version", default=False, action="store_true",
                               help="prints the version of the library and exits")
        argparser.add_argument("-l", "--log", default=None, type=str,
                               help="path to the top-level log file (default: None)")
        argparser.add_argument("-v", "--verbose", action="count", default=0,
                               help="set logging terminal verbosity level (additive)")
        argparser.add_argument("--silent", action="store_true", default=False,
                               help="deactivates all console logging activities")
        argparser.add_argument("--force-stdout", action="store_true", default=False,
                               help="force logging output to stdout instead of stderr")
        subparsers = argparser.add_subparsers(title="Operating mode", dest="mode")

        for app_mode in self.apps_table.keys():
            self.apps_table[app_mode]["parser_func"](subparsers, app_mode, argparse)

        return argparser

    def main(self, args=None):
        """Main entrypoint to use with console applications.

        This function parses command line arguments and dispatches the execution based on the selected
        operating mode. Run with ``--help`` for information on the available arguments.
        """
        args = self.setup(args=args)

        if isinstance(args, int):
            return args  # CLI must exit immediately with provided error code

        self.apps_table[args.mode]["application_func"](args)

        return 0


if __name__ == "__main__":
    mainApp = Application()
    mainApp.main()
