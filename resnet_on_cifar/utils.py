import subprocess


def get_git_hash():
  try:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   stderr=subprocess.STDOUT).decode()
  except subprocess.CalledProcessError as e:
    return "{}\n".format(e.output.decode("utf-8"))


def get_git_diff():
  try:
    return subprocess.check_output(['git', 'diff'],
                                   stderr=subprocess.STDOUT).decode()
  except subprocess.CalledProcessError as e:
    return "{}\n".format(e.output.decode("utf-8"))


class Logger(object):
  def __init__(self, stream, log_file):
    self.stream = stream
    self.log = log_file

  def write(self, msg):
    self.stream.write(msg)
    self.log.write(msg)

  def flush(self):
    self.stream.flush()
    self.log.flush()
