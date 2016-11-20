 
from __future__ import print_function

import os
import time
import sys

# Following the advice here:
# https://www.quora.com/How-can-I-delete-the-
#   last-printed-line-in-Python-language
ERASE_LINE = '\x1b[2K'

def clean():
  sys.stdout.write(ERASE_LINE)
  sys.stdout.write("\r")
  sys.stdout.flush()

def cleanwrite(m):
  clean()
  sys.stdout.write(m)
  sys.stdout.flush()

def progress(p, message):
  """
  p is a float between zero and one.
  """
  r, c = os.popen('stty size', 'r').read().split()
  c = int(c)
  pad = 4
  totalwidth = c-pad
  fstring = "\r{{:<{}s}}".format(totalwidth)
  fstring += " "*pad
  
  messagewidth = len(message) + 1
  barwidth = totalwidth - messagewidth - 2
  progresswidth = int(float(p)*barwidth)
  zeroswidth = barwidth - progresswidth
  bar = "".join([
    "[",
    "#"*progresswidth,
    " "*zeroswidth,
    "]",
    ])  
  m = fstring.format(message + " " + bar)
  cleanwrite(m)
  
def make_tests():
  i = 0
  N = 300
  while True:
    p = 1.0*(i%N) / (N-1)
    time.sleep(0.01)
    message = str(i)
    progress(p, message)
    i += 1

if __name__ == "__main__":
  make_tests()
