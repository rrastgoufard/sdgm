 
from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle
import os
from collections import defaultdict
from helpers import load_data, mkdir, floatX

epsilon = T.constant(3e-8)

def softmax(x, epsilon=epsilon):
  """
  Redefine the softmax here.  Copy code on
  theano's website.  The standard T.nnet.softmax
  function does not fit our Nth order tensors.
  """
  ex = T.exp(x - T.max(x, axis=-1, keepdims=True))
  sm = ex / T.sum(ex, axis=-1, keepdims=True)
  sm += epsilon
  sm /= T.sum(sm, axis=-1, keepdims=True)
  return sm

NLNames = defaultdict(lambda: T.nnet.sigmoid)
NLNames["sigmoid"] = T.nnet.sigmoid
NLNames["linear"] = lambda x: x
NLNames["softmax"] = softmax  
NLNames["softplus"] = T.nnet.softplus
NLNames["rectify"] = T.nnet.relu

class Stack(object):
  
  def __init__(self,
               insizes,
               outsizes,
               hidsize=500,
               hidnls=["rectify"]*2,
               lastnl="linear",
               ):
    """
    Takes a list of input sizes, a list of
    output sizes, hidden layer size 
    (all hidden layers are assumed to have the
    same output size),
    hidden layer nonlinearities, and output
    layer nonlinearity.
    
    All inputs are ``concatenated'' after
    passing through the first layer.  There
    is one weight matrix allocated per input.
    The second to last layer is expanded to
    however many outputs there are.
    """
    hidsizes = [hidsize]*len(hidnls)
    self.insizes = insizes
    self.outsizes = outsizes
    self.hidsizes = hidsizes
    self.hidnls = hidnls
    self.lastnl = lastnl
    
    self.params = []
    
    self.sizes = [insizes] + hidsizes + [outsizes]
    self.nls = hidnls + [lastnl]
    
    self.Ws = []
    self.bs = []
    self.ss = []
    
    def maybeglorot(nl, sin, sout):
      if nl in ["rectify","softplus"]:
        sd = np.sqrt(2.0/(sin+sout))
        if nl in ["softplus"]:
          sd /= 2
        return sd        
      return 0.01
    
    i = 0    
    for sin, sout, nl in zip(self.sizes, 
                             self.sizes[1:], 
                             self.nls):
      # If first iteration, then we have multiple
      # inputs to deal with.
      if i == 0:
        WIs = []
        for j, s0 in enumerate(sin):
          sd = maybeglorot(nl, s0, sout)
          W = floatX(np.random.randn(s0,sout)*sd)
          W = theano.shared(W, name="WI{}".format(j))
          WIs.append(W)
          self.params.append(W)
        self.Ws.append(WIs)
        
      # If last iteration, then we have multiple
      # outputs.  Note that each output can have
      # its own bias vector.
      elif i == len(self.sizes)-2:
        WOs = []
        bOs = []
        for j, so in enumerate(sout):
          sd = maybeglorot(nl, sin, so)
          W = floatX(np.random.randn(sin,so)*sd)
          W = theano.shared(W, name="WO{}".format(j))
          WOs.append(W)
          self.params.append(W)
          b = floatX(np.zeros(so))
          b = theano.shared(b, name="bO{}".format(j))
          bOs.append(b)
          self.params.append(b)
        self.bs.append(bOs)
        self.Ws.append(WOs)
      
      # All layers in the middle are easy.
      else:
        sd = maybeglorot(nl, sin, sout)
        W = floatX(np.random.randn(sin,sout)*sd)
        W = theano.shared(W, name="W{}".format(i))
        self.params.append(W)
        self.Ws.append(W)
      
      # If not the last layer, then create
      # a bias vector for this layer.
      if not i == len(self.sizes)-2:
        b = floatX(np.zeros(sout))
        b = theano.shared(b, name="b{}".format(i))
        self.params.append(b)
        self.bs.append(b)
      
      s = NLNames[nl]
      self.ss.append(s)
      
      i += 1
    
  #$ stack
  def __call__(self, Xs):
    for i, (W, b, s) in enumerate(zip(self.Ws, 
                                      self.bs, 
                                      self.ss)):
      
      # If first layer, deal with multiple inputs.
      if i == 0:
        XWI = sum(T.dot(X,WI) 
                  for X,WI in zip(Xs,W))
        X = s(XWI + b)
      
      # If last layer, deal with multiple outs.
      elif i == len(self.Ws) - 1:
        X = [s(T.dot(X,WO) + bO)
             for WO,bO in zip(W,b)]
      else:
        X = s(T.dot(X, W) + b)
        
    return X[0] if len(X) == 1 else X
  #$
  
  def save(self, fname):
    with open(fname, "wb") as p:
      vs = [np.array(param.eval()) 
            for param in self.params]
      pickle.dump([self.sizes,self.nls,vs],p)
  
  def load(self, fname):
    if not os.path.exists(fname):
      return False
    
    with open(fname, "rb") as p:
      params = pickle.load(p)
      
    [self.sizes,self.nls,values] = params
    for param, value in zip(self.params, values):
      param.set_value(value)
    
    self.ss = [NLNames[nl] for nl in self.nls]
    return True
    
  def __repr__(self):
    s = "{} {}".format(self.sizes,self.nls)
    return s
  
class Model(object):
  """
  Empty class to hold random bits and pieces to
  be passed around.
  
  A built model must have been
  assigned an objective and params.
  Also, it must have been assigned a set of
  input variables.
  """
  def __init__(self, name="", 
               shuffledata=False,
               thresholddata=None,
               normalizedata=False,
               seed=None,
               maxvar=None):
    self.name = name
    self.shuffledata = shuffledata
    self.thresholddata = thresholddata
    self.load_data(NSamples=10, 
                   shuffledata=shuffledata,
                   thresholddata=thresholddata,
                   normalizedata=normalizedata,
                   seed=seed,
                   maxvar=maxvar)
    self.inputs = None
    self.objective = None
    self.params = None
    self.adds = []
    self._trainer = None
    self.descenter = None
    self.Print = None
    self.savemomentum = False
    
    # Keep track of how many data points we have
    # drawn in each shuffle.
    self.Su = 1e30
    self.Sl = 1e30
    
  def load_data(self, **kwargs):
    self.data = load_data(**kwargs)
    self.XCols = np.sum(self.data.idxcols)
    self.seed = self.data.seed
    
    #$ load_data
    XU, XL, XT = [
      theano.shared(floatX(xx)) for xx in [
        self.data.train_x,
        self.data.small_x,
        self.data.test_x,]]
    YL, YLh, YT = [
      theano.shared(floatX(xx)) for xx in [
        self.data.small_y,
        self.data.small_yh,
        self.data.test_y,]]
    self.Xu = XU
    self.Xl = XL
    self.Yl = T.cast(YL, "int32")
    self.Ylh = YLh
    self.Xt = XT
    self.Yt = T.cast(YT, "int32")
    #$
    
  def log(self, *args):
    if self.Print is not None:
      self.Print(*args)
        
  def save(self, savedir="../params"):
    mkdir(savedir)
    for name, net in self.networks.iteritems():
      net.save(os.path.join(savedir,name + ".pkl"))
    if self.descenter is not None:
      # Always save the momentum.
      self.descenter.save(savedir)
      
  def load(self, loaddir="../params"):
    for name, net in self.networks.iteritems():
      net.load(os.path.join(loaddir,name + ".pkl"))
    if self.descenter is not None:
      if self.loadmomentum:
        # Reload the momentum only if
        # self.loadmomentum is True.
        self.descenter.load(loaddir)
      else:
        self.descenter.reset()
  
  def reset(self):
    self.descenter.reset()
    
  def maketrainer(self):
    """
    In order to make a trainer function, the model
    needs to have been assigned an objective,
    params, inputs, adds, and a descenter.
    """
    if self.descenter is None:
      raise NotImplementedError(
        "Need to assign a descenter first...")
    
    updates = self.descenter(
      self.objective, self.params, 
      )
    self.log(self.descenter.settings)
    self.log("Created shareds for", self.descenter)
    self.log("Beginning trainer compilation")
    self._trainer = theano.function(
      inputs=self.inputs,
      outputs=[self.objective],
      updates=updates,
      #profile=True,
      allow_input_downcast=True)
    self.log("Done trainer compilation!")
  
  def step(self, *args, **kwargs):
    if self._trainer is None:
      raise NotImplementedError(
        "Need to call model.maketrainer() first!")
    return map(float, self._trainer(*args, **kwargs))
