from __future__ import print_function

import progress
import os
from helpers import Log
import numpy as np

class SaveJuggler(object):
  def __init__(self, 
               model,
               NSaves=2, 
               combolength=30,
               savedir="../params",
               jugglemomentum=True,
               enablesave=True,
               ):
    self.model = model
    self.NSaves = NSaves
    self.enablesave = enablesave
    self.savedir = savedir
    self.combolength = combolength
    self.jugglemomentum = jugglemomentum
    self.safe = 0
    self.next_save = 0
    self.newsafe = False
    
    self.combo = 0
    self.failures = 0
    self.wasted = 0
    self.potential_wastes = 0
    self.saves = 0
    self.sincesave = 0
    self.sinceload = 0
    
    self.Print = None
    self.messagequeue = []
    
  def tick(self, mul=None):
    mul = self.NSaves if mul is None else mul
    self.newsafe = False
    self.combo += 1
    self.potential_wastes += 1
    self.sincesave += 1
    self.sinceload += 1
    if self.iscombo:
      if self.combo >= mul*self.combolength:
        self.sincesave = 0
        self.newsafe = True
        self.saves += 1
        self.safe = (self.safe+1) % self.NSaves
        self.potential_wastes = 0
        self.log("Safety to", self.safe, self.saves)
      self._save()
        
  def _save(self):
    savedir = os.path.join(self.savedir,
      "save{}".format(self.next_save))
    self.log("Saving to", savedir, 
             self.safe, self.combo)
    if self.enablesave:
      self.model.save(savedir)
      self.next_save = self.next_save + 1
      self.next_save = self.next_save % self.NSaves
  
  def fail(self):
    self.wasted += self.potential_wastes + 1
    self.failures += 1
    self.next_save = (self.safe+1) % self.NSaves
    loaddir = os.path.join(self.savedir,
      "save{}".format(self.safe))
    self.log("Loading from", loaddir, 
             self.safe, self.sincesave)
    self.combo = 0
    self.potential_wastes = 0
    self.sinceload = 0
    self.model.load(loaddir)
    if self.jugglemomentum:
      lm = self.model.loadmomentum
      self.model.loadmomentum = not lm
      self.log("load momentum set to",
               self.model.loadmomentum)
    self.messagequeue = []
      
  def cleanstart(self, logging=False):
    if logging:
      self.Print = Log("../log/savejuggler","w",
                       quiet=True)
    self._save()
    
  def log(self, *args, **kwargs):
    if self.Print is not None:
      self.Print(*args, **kwargs)
  
  @property
  def iscombo(self):
    if self.combo > 0:
      return (self.combo % self.combolength) == 0
    return False
  
  def inject(self, message, qlen=None):
    qlen = self.NSaves if qlen is None else qlen
    self.messagequeue.insert(0, message)
    if len(self.messagequeue) == qlen:
      m = self.messagequeue.pop()
      progress.clean()
      self.model.log(m)
    
  def p(self, epoch, mb, NBatches, mul=None):
    pstring = "{:>03d} {:>05d}/{:>05d}, "
    #pstring += "{} since save.  "
    #pstring += "{} since load.  "
    #pstring += "{} loaded, "
    pstring += "{} saved.  "
    #if self.model.loadmomentum:
      #pstring += "Restore momentum."
    #else:
      #pstring += "Discard momentum."
    
    pstring += "  Success Combo?!? -->"
    mul = self.NSaves if mul is None else mul
    den = self.combolength * mul
    if self.combo >= den:
      den = self.combolength
    num = self.combo % den
    if num == 0:
      p = 1.0
    else:
      p = 1.0*num / den
    
    progress.progress(p,
      pstring.format(epoch, mb, NBatches,
                     #self.sincesave,
                     #self.sinceload,
                     #self.failures, 
                     self.saves,
                     ))

class LinearJuggler(SaveJuggler):
  pass

class RandomJuggler(SaveJuggler):
  def tick(self):
    super(RandomJuggler, self).tick(mul=1)
  def inject(self, *args):
    super(RandomJuggler, self).inject(*args, qlen=1)
  def fail(self):
    self.safe = np.random.choice(
      range(min(self.NSaves, self.saves)))
    super(RandomJuggler, self).fail()
  def p(self, *args):
    super(RandomJuggler, self).p(*args, mul=1)
