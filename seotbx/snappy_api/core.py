import logging
import seotbx
import snappy
import os
logger = logging.getLogger("seotbx.snappy_api.apps.sentinel1")

def run_gpt_base(command_args, version=1):
    gpt_bin = os.path.join(os.environ['SNAP_HOME'], 'bin/gpt')
    if not os.path.exists(gpt_bin):
        raise Exception(f"gpt not found: {gpt_bin}")
    command_args.insert(0, gpt_bin)
    if version == 1:
        command = ' '.join(command_args)
        logging.info(command)
        graph_thread = seotbx.snappy_api.threads.RunCommand(command=command)
    else:
        command = command_args
        graph_thread = seotbx.snappy_api.threads.RunCommand2(command=command)
        logging.info(command)
    graph_thread.start()
    graph_thread.join()