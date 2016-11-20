
from __future__ import print_function
import os, gzip, errno, datetime, time
import cPickle as pickle
import numpy as np
import scipy
import theano

from collections import namedtuple

datafile = os.path.join(
  "../given_files",
  "mnist.pkl.gz",
  )

def floatX(X):
  return np.array(X, dtype=theano.config.floatX)

def mkdir(location):
  try:
    os.makedirs(location)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
    
def rm(location):
  try:
    os.remove(location)
  except OSError as exception:
    if exception.errno != errno.ENOENT:
      raise
    
def onehot(y, n):
  N = y.shape[0]
  hot = np.zeros([N,n])
  hot[np.arange(N), y] = 1
  return floatX(hot)

class Log(object):
  """
  A simple logger that appends to a log file
  whenever a message is to be written.  The
  logger automatically prepends the time that
  each message is committed to the log file.
  """
  def __init__(self, fname, 
               mode="a", quiet=False):
    dirname = os.path.dirname(fname)
    filename = os.path.relpath(fname, dirname)
    timestr = "_".join(["{:>02d}".format(t) 
      for t in time.localtime()[:6]])
    filename = timestr + "__" + filename
    fullname = os.path.join(dirname, filename)
    mkdir(dirname)
    
    self.fout = open(fullname + ".txt", mode)
    self.first_write = True
    self.fmt = "%Y_%m_%d %H:%M:%S"
    self.tstart = time.time()
    self.quiet = quiet
    
  def p(self, *message):
    joined = " ".join(map(str,message))
    if self.first_write:
      self.fout.write("\n")
      self.first_write = False
    for line in joined.split("\n"):
      d = datetime.datetime.now()
      self.fout.write("{} {:>14.7f} :: {}\n".format(
        d.strftime(self.fmt),
        time.time() - self.tstart,
        line))
    self.fout.flush()
    if not self.quiet:
      print(joined)
    
  def __call__(self, *args, **kwargs):
    self.p(*args, **kwargs)
    
def load_data(datafile=datafile, 
              classes=[], 
              NSamples=10,
              shuffledata=False,
              thresholddata=None,
              normalizedata=False,
              seed=None,
              maxvar=None):
  """
  Load the data.  
  The mnist.pkl.gz dataset
  contains three sets of data -- a pair
  of (x,y) training data, a pair for 
  validation, and a pair for testing.
  We ignore the validation set.
  """
  datasets = pickle.load(gzip.open(datafile,"rb"))
  train_x, train_y = datasets[0]
  test_x, test_y = datasets[2]
  
  if shuffledata:
    if seed is None:
      seed = np.random.randint(low=0,high=999999)
    rng = np.random.RandomState(seed)  
    idxs = rng.permutation(
      np.arange(train_x.shape[0]))
    train_x = train_x[idxs]
    train_y = train_y[idxs]
    
  if normalizedata:
    train_x -= np.min(train_x)
    train_x /= np.max(train_x)
    test_x -= np.min(test_x)
    test_x /= np.max(test_x)
  
  # If maxvar is specified, then filter the data
  # columns to include only those that have 
  # variance larger than maxvar.
  if maxvar is None:
    maxvar = 0
  idxcols = np.var(train_x, axis=0) >= maxvar
  train_x = train_x[:,idxcols]
  test_x = test_x[:,idxcols]
  
  if thresholddata is not None:
    t = thresholddata
    train_x = np.ones_like(train_x) * train_x > t
    test_x = np.ones_like(test_x) * test_x > t
    
  
  # Make a small labeled set that uses
  # NSamples samples from the specified classes
  if not classes:
    classes = list(range(10))
  
  idx = np.hstack([
    np.array(np.where(train_y == c)[0][:NSamples])
    for c in classes])
  small_x = train_x[idx]
  small_y = train_y[idx]
  
  small_y01 = np.array(small_y)
  for i, c in enumerate(classes):
    small_y01[small_y==c] = i
  small_yh = onehot(small_y01, len(classes))
  
  Data = namedtuple("Data",
    "train_x train_y test_x test_y " + 
    "small_x small_y small_y01 small_yh " + 
    "idxcols seed")
  data = Data(*[var for var in [
      train_x, train_y, test_x, test_y,
      small_x, small_y, small_y01, small_yh,
      idxcols, seed,
      ]]) 
  
  return data
