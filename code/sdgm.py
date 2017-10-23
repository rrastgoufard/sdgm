 
from __future__ import print_function

import os
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

#theano.config.profiling.ignore_first_call=True

import progress
from neuralnetwork import Model, Stack
import gradients as G
from helpers import Log, onehot, floatX
from savejuggler import LinearJuggler, RandomJuggler

#$ dimensions
Nl = 100  # number of labeled data points in batch
Nu = 100  # number of unlabeled data points

X = 784   # dimension of data.  If highvaronly
          # is specified, then this is 
          # overwritten.

Y = 10    # number of classes
Z = 88    # dimension of latent space
A = 99    # dimension of auxiliary variable
L = 1     # number of integration samples for Z
K = 1     # number of integration samples for A
Kt = 25   # number of MC samples for testing.
          #   Only A and X are involved in making
          #   class predictions, not Z.
#$

ADGM = False          # ADGM?  False means SDGM.

AtoZ = False          # Connect A to Z in
                      # inference model?
                      # This has no meaning for
                      # ADGM = False.
                      # Maaloe's first paper 
                      # has it disconnected.
AtoZ = AtoZ if ADGM else True

Anormal = True        # Determines whether P(A)
                      # is standard normal.
Anormal = Anormal if ADGM else False

gaussianX = False     # For MNIST, x is binary.
sampleX = True        # Do we sample it to make
                      # sure it takes values
                      # only at zero and one?
                      # (Grayscale values will
                      # be randomized every 
                      # training step.
                      
thresholdX = None     # Do we threshold it so
                      # that grayscale values
                      # are forced to zero/one?
                      
normalizeX = True     # Do we normalize X to be
                      # have minimum at zero
                      # and max at 1?

highvaronly = None    # Choose only the columns
                      # of X that have variance
                      # larger than this value.
                      # Set to None to use all
                      # columns.

shuffledata = True    # Shuffle the data or use
                      # the original 100 samples
                      # for consistency?

seed = None           # Specify a seed for
                      # generating the shuffled
                      # data.
#seed = 392422

gradnorm = False      # Whether or not to normalize
                      # the gradient norm.

loadmomentum = True     # Do we load the momentum
                        # of the descenter after
                        # a failure?

jugglemomentum = False  # After every failure,
                        # we switch whether or
                        # not to restore the 
                        # momentum of the last
                        # save.

randomjuggler = False   # Force linear save history
                        # or allow random movement?

nepochs = 25
combolength = 100       # savejuggler parameters...
NSaves = 2              # NSaves is a measure of
                        # paranoia or how badly
                        # we expect to get stuck.
enablesave = False

aJL = 1        # How much should JL be weighted?
aJU = 1        # How much should JU be weighted?
aJA = 50       # How much should JA be weighted?
aJW = 0        # How much weight regularization?

# Some consts to help avoid float64 stuff.
zero = T.constant(0)
one = T.constant(1)
two = T.constant(2)
half = T.constant(0.5)
twopi = T.constant(2*np.pi)

#epsilon = T.constant(1e-6)
    # 1e-6 is what maaloe's bernoulli uses
    # to clip mu and 1-mu.
epsilon = T.constant(3e-8)    
    # 3e-8 is the magic number.
    # With float32, 1 - 1e-8 is equal to 1.
    #               1 - 3e-8 is not equal to 1.
  
AxisN = 0 # Enums that list which variables 
AxisA = 1 # go into which axes.  
AxisZ = 2 
AxisY = 3 
AxisX = 4 

def makemodel(
    name="ADGM" if ADGM else "SDGM",
    nls=["rectify"]*2,
    seed=seed,
    descenter=G.Adam,
    K=K,
    L=L,
    ):
  #$ tensor_shapes
  """
  Creates the ADGM or SDGM model.

  Xl has dimension (Nl, 1, 1, 1, X)
  Xu has dimension (Nu, 1, 1, 1, X)
  Yl has dimension (Nl, 1, 1, 1, Y)
  Yu has dimension ( 1, 1, 1, Y, Y)
  EAl has dimension (Nl, K, 1, 1, A)
  EAu has dimension (Nu, K, 1, 1, A)
  EZl has dimension (Nl, K, L, 1, Z)
  EZu has dimension (Nu, K, L, Y, Z)
  Al will have dimension (Nl, K, 1, 1, A)
  Au will have dimension (Nu, K, 1, 1, A)
  Zl will have dimension (Nl, K, L, 1, Z)
  Zu will have dimension (Nu, K, L, Y, Z)
  """
  #$
  
  Print = Log("../log/{}".format(name),"w",
              quiet=True)
  model = Model(name=name, 
                shuffledata=shuffledata,
                thresholddata=thresholdX,
                normalizedata=normalizeX,
                seed=seed,
                maxvar=highvaronly)
  model.Print = Print
  model.loadmomentum = loadmomentum
  model.descenter = descenter(gradnorm)
  networks = OrderedDict()
  rng = MRG_RandomStreams()
  
  X = model.XCols
  
  model.constants = OrderedDict([
    ("                    ", model.name),
    ("shuffle data?", shuffledata),
    ("data seed", model.seed),
    ("Nu", Nu),
    ("Nl", Nl),
    ("X", X),
    ("Y", Y),
    ("Z", Z),
    ("A", A),
    ("L", L),
    ("K", K),
    ("Kt", Kt),
    ("aJL", aJL),
    ("aJU", aJU),
    ("aJA", aJA),
    ("aJW", aJW),
    ("gradient norm?", gradnorm),
    ("std. normal A?", Anormal),
    ("A to Z?", AtoZ),
    ("gaussian X?", gaussianX),
    ("sample X?", sampleX),
    ("threshold X?", thresholdX),
    ("normalize X?", normalizeX),
    ("high var only?", highvaronly),
    ("NSaves", NSaves),
    ("enable save?", enablesave),
    ("combolength", combolength),
    ("load momentum?", loadmomentum),
    ("juggle momentum?", jugglemomentum),
    ("random juggler?", randomjuggler),
    ("epsilon", epsilon),
    ])
  for name, val in model.constants.items():
    model.Print("{:>20s}".format(name), val)  
  
  #$ px_stack
  # Create the networks for px
  ins = [Y,Z] if ADGM else [A,Y,Z]
  last = "linear" if gaussianX else "sigmoid"
  O = [X,X] if gaussianX else [X]
  fx = Stack(insizes=ins, outsizes=O,
             hidnls=nls, lastnl=last)
  networks["fx"] = fx
  #$
  
  #$ pa_stack
  # Create the networks for pa
  ins = [X,Y,Z] if ADGM else [Y,Z]
  fa = Stack(insizes=ins, outsizes=[A,A],
             hidnls=nls)
  if not Anormal:
    networks["fa"] = fa
  #$
  
  #$ qz_stack
  # Create the networks for qz
  ins = [A,X,Y] if AtoZ else [X,Y]
  fz = Stack(insizes=ins, outsizes=[Z,Z],
             hidnls=nls)
  networks["fz"] = fz
  #$
  
  #$ qax_stack
  # Create the networks for qax
  ins = [X]
  fax = Stack(insizes=ins, outsizes=[A,A],
              hidnls=nls)
  networks["fax"] = fax
  #$
  
  #$ qy_stack
  # Create the network for qy.  Outputs are
  # probabilities, so last layer is always 
  # softmax.
  ins = [A,X]
  last = "softmax"
  fy = Stack(insizes=ins, outsizes=[Y],
             hidnls=nls, lastnl=last)
  networks["fy"] = fy
  #$
  
  #$ model.networks
  # Collect all of the parameters together
  # so we can optimize the objectives with
  # respect to them.
  model.networks = networks
  model.params = []
  for name, net in model.networks.items():
    model.Print("{:>20s}".format(name), net)
    model.params += net.params
  #$
  
  # For now, throw an error if Nl or Nu are 
  # not specified.
  # Eventually, we would like to be able to 
  # handle only Nl, only Nu, or both Nl and Nu.
  if Nl is None or Nu is None:
    raise ValueError("Need to specify Nl and Nu")
  
  #$ shared_inputs
  # Xl, Ylh, and Xu are shared variables on the
  # GPU.  For Xu, we take random batch slices.  
  # We assume for now that all (Xl,Yl) are used
  # in each batch.  
  Xl2 = model.Xl[:Nl]
  Yl2 = model.Ylh[:Nl]
  
  bidxs = rng.uniform((Nu,))*model.Xu.shape[0]
  bidxs = T.cast(bidxs, "int32")
  Xu2 = model.Xu[bidxs]
  #$
  
  #$ sampleX
  # If X is binary, then sample it on each
  # minibatch.  This idea borrowed from Maaloe's
  # code.  Not sure if it helps.
  # 
  # Keep track of Xl2s, Yl2, and Xu2s so we can
  # do theano variable substitution later.
  if not gaussianX and sampleX:
    Xl2s = rng.binomial(n=1,p=Xl2,size=Xl2.shape,
                        dtype=theano.config.floatX)
    Xu2s = rng.binomial(n=1,p=Xu2,size=Xu2.shape,
                        dtype=theano.config.floatX)
  else:
    Xl2s = Xl2
    Xu2s = Xu2
  #$
  
  #$ dimshuffled
  # Reshape the labeled set matrices 
  # to 5th-order tensors.
  Xl = Xl2s.dimshuffle([0,"x","x","x",1])
  Yl = Yl2.dimshuffle([0,"x","x","x",1])
  
  # Xu is known, but Yu is not known.  
  # Create one possible Y per class.
  Xu = Xu2s.dimshuffle([0,"x","x","x",1])
  Yu = T.eye(Y, Y).dimshuffle(["x","x","x",0,1])
  #$
  
  #$ noises
  # EZ and EA will be used to approximate 
  # the integrals using L samples for Z and
  # K samples for A.  
  # 
  # Create shared variables for K and L so we
  # can do variable substitutions later.
  K = theano.shared(K, name="samplesA")
  L = theano.shared(L, name="samplesZ")  
  EAl = rng.normal((Xl.shape[0], K, 1, 1, A))
  EAu = rng.normal((Xu.shape[0], K, 1, 1, A))
  EZl = rng.normal((Xl.shape[0], K, L, 1, Z))
  EZu = rng.normal((Xu.shape[0], K, L, Y, Z))
  #$
  
  # Assign inputs to the model.
  # We assume that all data is already on the GPU.
  # Furthermore, we create functions that
  # evaluate the objectives on the test data
  # directly.  Therefore, there are no inputs
  # needed for calling the training function.
  model.inputs = []
  
  #$ al_au
  # Find the latent variables.
  # Note that multiplying by E effectively tiles
  # all latent variables L or K times.
  # 
  # Auxiliary A has to be found first 
  # because latent Z is a function of it.
  muaxl, sdaxl = fax([Xl])
  muaxu, sdaxu = fax([Xu])
  Al = muaxl + T.exp(sdaxl)*EAl
  Au = muaxu + T.exp(sdaxu)*EAu
  #$
  
  #$ zl_zu
  # Compute Z.
  inputl = [Al,Xl,Yl] if AtoZ else [Xl,Yl]
  inputu = [Au,Xu,Yu] if AtoZ else [Xu,Yu]
  muzl, sdzl = fz(inputl)
  muzu, sdzu = fz(inputu)
  Zl = muzl + T.exp(sdzl)*EZl
  Zu = muzu + T.exp(sdzu)*EZu
  #$
  
  #$ muxl_muxu
  # Find the reconstruction means and 
  # standard deviations.  
  # Note: sdxl and sdxu are used only if
  #       gaussian is True.  The binary case
  #       ignores those.
  # If ADGM, then X is a function of YZ.
  # If SDGM, then X is a function of AYZ.
  inputl = [Yl, Zl] if ADGM else [Al,Yl,Zl]
  inputu = [Yu, Zu] if ADGM else [Au,Yu,Zu]
  if gaussianX:
    muxl, sdxl = fx(inputl)
    muxu, sdxu = fx(inputu)
  else:
    muxl = fx(inputl)
    muxu = fx(inputu)
  #$
  
  #$ mual_muau
  # Find mu and sd for A in the generative 
  # (reconstruction) direction.
  # If ADGM, then A depends on XYZ.
  # If SDGM, then A depends on YZ.
  inputl = [Xl,Yl,Zl] if ADGM else [Yl,Zl]
  inputu = [Xu,Yu,Zu] if ADGM else [Yu,Zu]
  mual, sdal = fa(inputl)
  muau, sdau = fa(inputu)
  #$

  #$ JL_1
  # Find the component probabilities and the
  # labeled objective, JL.
  l_pz = loggauss(Zl)
  l_qz = loggauss(Zl, muzl, sdzl)
  
  l_py = T.log(1.0/Y)
  
  if gaussianX:
    l_px = loggauss(Xl, muxl, sdxl)
  else:
    l_px = logbernoulli(Xl, muxl)
  #$
  
  #$ JL_2
  # In Maaloe's first revision, A is disconnected
  # in the generative model, so we assume it
  # to be standard normal.
  # 
  # In the more updated version, A is fed into
  # by X, Y, and Z.
  # In SDGM, A is generated by Z and Y.
  normal = zero if Anormal else one
  l_pa = loggauss(Al, normal*mual, normal*sdal)
  l_qa = loggauss(Al, muaxl, sdaxl)
  #$
  
  #$ JL_3
  JL = l_qz + l_qa
  JL = JL - l_px - l_py - l_pz - l_pa
  JL = batchaverage(exA(exZ(JL)))
  JL = aJL * JL
  #$
  
  #$ JU_1
  # Find the component probabilities and the
  # unlabeled objective, JU.
  
  # The output of fy(Au, Xu) is pi.
  # (Nu, K, 1, 1, Y)
  # We need to relocate the last axis.
  # (Nu, K, 1, Y, 1)
  inputu = [Au, Xu]
  pi = fy(inputu).dimshuffle([0,1,"x",4,"x"])
  #$
  
  #$ JU_2
  u_pz = loggauss(Zu)
  u_qz = loggauss(Zu, muzu, sdzu)
  
  u_py = T.log(1.0/Y)
  u_qy = T.log(pi)
  
  u_pa = loggauss(Au, normal*muau, normal*sdau)
  u_qa = loggauss(Au, muaxu, sdaxu)
  
  if gaussianX:
    u_px = loggauss(Xu, muxu, sdxu)
  else:
    u_px = logbernoulli(Xu, muxu)
  #$
  
  #$ JU_3
  JU = u_qz + u_qa + u_qy
  JU = JU - u_px - u_py - u_pz - u_pa
  JU = batchaverage(exA(classsum(exZ(JU), pi)))
  JU = aJU * JU
  #$
  
  #$ JA
  # Make sure that the known labels are correctly
  # assigned.
  # Yl has dimension (Nl, 1, 1, 1, Y)
  # Al,Xl has dimension (Nl, K, 1, 1, A+X)
  # fy(Al,Xl) is (Nl, K, 1, 1, Y)
  #
  # Yl is one-hot.
  # Multiply by Yl and perform a sum over 
  # Y to get the one probability out, then neg 
  # log it, average it over K, and 
  # average it over N.
  inputl = [Al, Xl]
  JA = batchaverage(exA(
    -T.log(T.sum(fy(inputl)*Yl, axis=-1))))
  JA = aJA * JA
  #$
  
  # Regularize the weight matrices of the
  # networks so they do not stray far from zero.
  # Copied from Maaloe's github code.
  JW = zero
  for p in model.params:
    if 'W' not in str(p):
      continue
    JW += T.mean(p**two)
  JW = aJW * JW
  
  JCombined = JL + JU + JA + JW

  # Stick the objectives into the model.
  model.objective = JCombined
  
  #$ prediction_comments
  # Create a function for predictions!
  # We need to evaluate a bunch of values for A,
  # so Xt is an N by X dimensional matrix and
  # Et is a K by A dimensional matrix.
  # Reshape Xt to (N, 1, X) and
  #         Et to (1, K, A).
  # 
  # Then, At = fmuax(Xt) + Et*fsdax(Xt)
  # and has a dimension of (N, K, A).
  # 
  # Class probabilities pi are fy(AXt)
  # and have shape (N, K, Y).  Take their
  # log, average over K, then argmax over Y
  # to find class predictions.
  #$
  
  #$ prediction_function
  Xt2 = T.matrix("Xt")
  Et2 = rng.normal((Kt,A))
  Xt = Xt2.dimshuffle([0,"x",1])
  Et = Et2.dimshuffle(["x",0,1])
  muat, sdat = fax([Xt])
  At = muat + T.exp(sdat)*Et
  
  inputt = [At, Xt]
  prediction = T.argmax(
    T.mean(T.log(fy(inputt)),axis=1), axis=-1)
  
  predict = theano.function(
    inputs=[Xt2],
    outputs=prediction,
    allow_input_downcast=True)
  model.predict = predict
  #$
  
  #$ classification
  Yt = T.ivector("Yt")
  accuracyT = T.eq(Yt,prediction).mean(
    dtype=theano.config.floatX)
  
  model.accuracyT = theano.function(
    inputs=[],
    outputs=accuracyT,
    givens={Xt2:model.Xt, Yt:model.Yt},
    allow_input_downcast=True)
  #$
  
  model.accuracyL = theano.function(
    inputs=[],
    outputs=accuracyT,
    givens={Xt2:model.Xl, Yt:model.Yl},
    allow_input_downcast=True)
  
  # Create a stats function that outputs 
  # extra information.
  model.adds = [JL, JU, JA, JW,
                T.mean(l_qa),
                T.mean(u_qa),
                T.mean(u_qy),
                T.mean(l_qz),
                T.mean(u_qz),
                -T.mean(l_px.max(axis=AxisY)),
                -T.mean(u_px.max(axis=AxisY)),
                -T.mean(l_pa),
                -T.mean(u_pa),
                ]
  model.headings = [
    "J",
    "JL",
    "JU",
    "JA",
    "JW",
    "l q(a)",
    "u q(a)",
    "u q(y)",
    "l q(z)",
    "u q(z)",
    "l -p(x)",
    "u -p(x)",
    "l -p(a)",
    "u -p(a)",
    ]
  model.outputs = [model.objective] + model.adds  
  model.stats = theano.function(
    inputs=[],
    outputs = model.outputs,
    givens={Xl2s:model.Xl[:1000],
            Yl2:model.Ylh[:1000],
            Xu2s:model.Xu[:1000],
            K:1},
    allow_input_downcast=True)
    
  return model

#$ batchaverage
def batchaverage(x):
  return T.mean(x)
#$

#$ classsum
def classsum(x, pi=None):
  """
  x has shape  (N, K, L, Y, X).
  pi has shape (N, K, 1, Y, 1) if given.
  """
  if pi is None:
    pi = one
  return T.sum(x*pi, axis=AxisY, keepdims=True)
#$

#$ exA_exZ
def exA(x):
  return T.mean(x, axis=AxisA, keepdims=True)

def exZ(x):
  return T.mean(x, axis=AxisZ, keepdims=True)
#$

#$ logbernoulli
def logbernoulli(x, mu):
  mu = T.clip(mu, epsilon, one-epsilon)
  cost = x*T.log(mu) + (one-x)*T.log(one-mu)
  return T.sum(cost, axis=AxisX, keepdims=True)
#$

#$ loggauss_comments
def loggauss(x, mu=0, s=0):
  """
  x is assumed to have shape (N, K, L, Y, X).
  All X dimensions are iid, so we can handle them
  separately and sum the results.  
  
  The covariance is assumed diagonal with
  elements exp(2*s).
  
  That is, the standard deviation is S = exp(s)
  such that variance is V = S*S = exp(2*s) and
  the standard deviation s is the log of S
  s = log(S).
  
  Return a set of (N, K, L, Y, 1) gaussian
  likelihoods.
  """  
#$
#$ loggauss_code
  V = T.exp(two*s)
  
  const = -half*T.log(twopi) - s
  exp = -half*(x-mu)**two/V
  return T.sum(const+exp,axis=AxisX,keepdims=True)
#$

def train(model, 
          nepochs=nepochs,
          combolength=combolength,
          NSaves=NSaves,
          maketrainer=True,
          ):
  
  if maketrainer:
    model.maketrainer()
  
  # Assuming that the unlabeled portion is
  # much larger than the labeled portion, then
  # there are len(model.data.train_x) / Nu
  # minibatches per epoch.
  NBatches = int(len(model.data.train_x) / Nu)
  
  # How many times we want to loop over the
  # entire unlabeled data set.
  # We want to have nepochs' worth of data in the
  # saved state.  We don't care about how much
  # was wasted.
  epoch = 0
  quittingtime = nepochs*NBatches/combolength
  model.Print("Will quit after", 
              quittingtime, "saves.")
  
  model.Print("Starting!")
  
  if randomjuggler:
    SJ = RandomJuggler
  else:
    SJ = LinearJuggler
  savejuggler = SJ(model, enablesave=enablesave,
    NSaves=NSaves, combolength=combolength,
    jugglemomentum=jugglemomentum)
  savejuggler.cleanstart()
  
  # Print a header for the data columns.
  headingstring = "{:>15s}  "*2
  headingstring += "{:>10s}  "*len(model.headings)
  vstring = "{:>10.3f}  "*len(model.headings)
  itstring = "{:>03d} {:>05d}/{:>05d}"
  
  header = headingstring.format(
    "Epoch mb/Mb   ",
    "{:>6s} {:>6s}".format("known", "test"), 
    *model.headings)
  model.Print(header)
  
  while savejuggler.saves < quittingtime:
    
    epoch += 1
    
    for mb in range(NBatches):

      v = np.sum(model.step())
      
      if np.isnan(v) or np.isinf(v):
        savejuggler.fail()
      else:
        savejuggler.tick()
        savejuggler.p(epoch, mb, NBatches)
      
      if savejuggler.iscombo:
        #model._trainer.print_summary()
        v = list(map(float, model.stats()))
        accuracyT = model.accuracyT()
        accuracyL = model.accuracyL()
        
        message = "{:>15s}  {:>15s}  {}".format(
          itstring.format(epoch, mb+1, NBatches),
          "{:>6.2f} {:>6.2f}".format(
            accuracyL*100, accuracyT*100), 
          vstring.format(*v))
        savejuggler.inject(message)
        
        if savejuggler.sincesave == 0:
          pstring = "{:>03d} {:>05d}/{:>05d} {}"
          print(pstring.format(
            epoch, mb+1, NBatches, accuracyT))
  
def testload(model):
  savedirs = ["../params/save{}".format(i) 
              for i in range(8)]
  for savedir in savedirs:
    if os.path.exists(savedir):
      model.load(savedir)
      
      for i in range(10):
        accuracyT = model.accuracyT()
        print("{:>6.2f} {}".format(
          accuracyT*100, savedir))
        
        
def trainmultiple(N=10):
  nldecs = [
    #(G.Adadelta, ["softplus"]*2),
    (G.Adam, ["rectify"]*2),
    ]
  
  for i in range(N):
    seed = np.random.randint(low=0,high=999999)
    for descenter, nls in nldecs:
      model = makemodel(seed=seed,
                        descenter=descenter,
                        nls=nls)
      train(model, nepochs=200)

if __name__ == "__main__":
  trainmultiple()
