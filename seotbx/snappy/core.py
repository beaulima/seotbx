import seotbx
import snappy
import os


def run_gpt_base(command_args, version=1):
    gpt_bin = os.path.join(os.environ['SNAP_HOME'], 'bin/gpt')
    if not os.path.exists(gpt_bin):
        raise Exception(f"gpt not found: {gpt_bin}")
    command_args.insert(0, gpt_bin)
    if version == 1:
        command = ' '.join(command_args)
        print(command)
        graph_thread = seotbx.snappy.threads.RunCommand(command=command)
    else:
        command = command_args
        graph_thread = seotbx.snappy.threads.RunCommand2(command=command)
        print(command)
    graph_thread.start()
    graph_thread.join()