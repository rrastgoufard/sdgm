We sometimes use adadelta with the softplus
nonlinearity.  This combination has a 
tendency to explode.  For this reason, we employ
``save juggling,'' described here. 

After 100 consecutive minibatches that are 
processed without exploding, we save all of the 
networks' parameters.

We create a buffer so that we always have a 
safe set of parameters.  A set of parameters
is considered safe if 100 minibatches were
processed after the set of parameters was
stored.  

If we did not create the buffer with a safe
save, then it would be possible for a save to
explode on the next minibatch regardless of 
the contents of the minibatch, and therefore this
set of parameters would be forever stuck.
