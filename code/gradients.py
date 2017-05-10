
from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
from helpers import floatX
from collections import OrderedDict

import os
import pickle

class Descenter(object):
  def __init__(self, gradnorm=False):
    self.variables = None
    self.gradnorm = gradnorm
    self.params = OrderedDict()
  
  def grad(self, *args, **kwargs):
    """
    Make sure the norm of all gradients is
    always fixed to maxnorm = 5.
    See lasagne total_norm_constraint.
    """
    grads = T.grad(*args, **kwargs)
    if self.gradnorm:
      norm = T.sqrt(sum(T.sum(g**2) 
                        for g in grads))
      tnorm = T.clip(norm, 0, 5)
      mul = tnorm / (1e-7+norm)
      grads = [T.clip(g*mul, -1, 1) 
               for g in grads]
    return grads

  def savename(self, savedir):
    return os.path.join(savedir,"descenter.pkl")
  
  def save(self, savedir):
    vals = [np.array(v.get_value()) 
            for v in self.variables]
    with open(self.savename(savedir), "w") as p:
      pickle.dump(vals, p)
      
  def load(self, loaddir):
    if not os.path.exists(self.savename(loaddir)):
      return False
    with open(self.savename(loaddir), "r") as p:
      vals = pickle.load(p)
    if self.variables is not None:
      for v, val in zip(self.variables, vals):
        v.set_value(val) 
    return True
  
  def reset(self):
    for v in self.variables:
      v.set_value(np.zeros_like(v.get_value()))
      
  def __repr__(self):
    return self.params["Descenter"]
  
  @property
  def settings(self):
    return "\n".join(["{:>20s} {}".format(k,v)
      for k, v in self.params.items()])
    
class RMSprop(Descenter):
  
  def __call__(self,
               cost, 
               params, 
               lr=0.001, 
               rho=0.9, 
               epsilon=1e-6,
               ):
    """
    RMSprop taken from youtube video's code.
    https://www.youtube.com/watch?v=S75EdAcXHKk
    """
    self.params = OrderedDict([
      ("Descenter", "RMSprop"),
      ("lr", lr),
      ("rho", rho),
      ("epsilon", epsilon),
      ])
    
    grads = self.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
      acc = theano.shared(
        np.zeros_like(p.get_value()),
        name=p.name + "p")
      acc_new = rho * acc + (1 - rho) * g ** 2
      gradient_scaling = T.sqrt(acc_new + epsilon)
      g = g / gradient_scaling
      updates.append((acc, acc_new))
      updates.append((p, p - lr * g))
    return updates     

class Adadelta(Descenter):  
  def __call__(self, 
               cost, 
               params, 
               rho=0.95,
               eps=1e-6,
               ):
    """
    Taken from 
    http://blog.wtf.sg/2014/08/28/
      implementing-adadelta/
      
    rho's default value in lasagne is 0.95.
    All of my experiments up until now have 
    used 0.9.
    """
    self.params = OrderedDict([
      ("Descenter", "Adadelta"),
      ("rho", rho),
      ("epsilon", eps),
      ])
    
    one = T.constant(1)
    gradients = self.grad(cost=cost, wrt=params)
    
    # create variables to store intermediate updates
    gradients_sq = [theano.shared(
      np.zeros_like(p.get_value()),
      name=p.name + "gsq") for p in params]
    deltas_sq = [theano.shared(
      np.zeros_like(p.get_value()),
      name = p.name + "dsq") for p in params]
    self.variables = []
    self.variables += gradients_sq
    self.variables += deltas_sq
  
    # calculates the new "average" delta for 
    # the next iteration
    gradients_sq_new = [ 
      rho*g_sq + (one-rho)*(g**2) 
      for g_sq,g 
      in zip(gradients_sq, gradients) ]
  
    # calculates the step in direction. 
    # The square root is an approximation to 
    # getting the RMS for the average value.
    deltas = [ 
      (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad 
      for d_sq,g_sq,grad 
      in zip(deltas_sq,gradients_sq_new,gradients) ]
  
    # calculates the new "average" deltas 
    # for the next step.
    deltas_sq_new = [ rho*d_sq + (one-rho)*(d**2) 
      for d_sq,d in zip(deltas_sq,deltas) ]
  
    # Prepare it as a list f
    gradient_sq_updates = zip(
      gradients_sq,gradients_sq_new)
    deltas_sq_updates = zip(
      deltas_sq,deltas_sq_new)
    parameters_updates = [ 
      (p,p - d) for p,d in zip(params,deltas) ]
    return gradient_sq_updates + \
      deltas_sq_updates + \
      parameters_updates  

class Adam(Descenter):
  """
  Taken from lasagne's updates code.
  """
  def __call__(self,
               cost, 
               params, 
               
               #learning_rate=0.001, 
               #beta1=0.9,
               #beta2=0.99,
               
               learning_rate=3e-4,
               beta1=0.9,
               beta2=0.999, 
               
               #learning_rate=3e-3,
               #beta1=0.9,
               #beta2=0.9999,
               
               epsilon=1e-8,
               ):
    """
    default learning_rate from lasagne is 0.001.
    Maaloe suggests 3e-4.
    """
    self.params = OrderedDict([
      ("Descenter", "Adam"),
      ("lr", learning_rate),
      ("beta1", beta1),
      ("beta2", beta2),
      ("epsilon", epsilon),
      ])
    
    all_grads = self.grad(cost=cost, wrt=params)
    t_prev = theano.shared(floatX(0.))
    updates = OrderedDict()

    # Using theano constant to prevent 
    # upcasting of float32.
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(one-beta2**t) / \
                              (one-beta1**t)

    self.variables = [t_prev]
    for param, g_t in zip(params, all_grads):
      value = param.get_value(borrow=True)
      m_prev = theano.shared(
        np.zeros(value.shape, dtype=value.dtype),
        broadcastable=param.broadcastable)
      v_prev = theano.shared(
        np.zeros(value.shape, dtype=value.dtype),
        broadcastable=param.broadcastable)
      self.variables += [m_prev, v_prev]

      m_t = beta1*m_prev + (one-beta1)*g_t
      v_t = beta2*v_prev + (one-beta2)*g_t**2
      step = a_t*m_t/(T.sqrt(v_t) + epsilon)

      updates[m_prev] = m_t
      updates[v_prev] = v_t
      updates[param] = param - step

    updates[t_prev] = t
    return updates
