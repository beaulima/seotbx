import logging
import threading
import subprocess

logger = logging.getLogger(__name__)

class RunCommand(threading.Thread):
    def __init__(self, command):
        self.stdout = None
        self.stderr = None
        self.command = command
        threading.Thread.__init__(self)

    def run(self):
        p = subprocess.Popen(self.command.split(),
                             shell=False,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             universal_newlines=True)
        #self.stdout, self.stderr  = p.communicate()

        for stdout_line in iter(p.stdout.readline, ""):
            logger.info(stdout_line.strip())
        p.stdout.close()
        return p.wait()


class RunCommand2(threading.Thread):
    def __init__(self, command):
        self.stdout = None
        self.stderr = None
        self.command = command
        threading.Thread.__init__(self)

    def run(self):
        p = subprocess.Popen(self.command,
                             shell=False,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             universal_newlines=True)
        #self.stdout, self.stderr  = p.communicate()

        for stdout_line in iter(p.stdout.readline, ""):
            logger.info(stdout_line.strip())
        p.stdout.close()
        return p.wait()
