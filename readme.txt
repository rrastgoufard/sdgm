
This is an implementation of a
Skip Deep Generative Model (SDGM) and has
flags that allow it to change to Auxiliary
Deep Generative Models (ADGM), both proposed
by Maaloe, Sonderby, Sonderby, and Winther.
https://arxiv.org/abs/1602.05473

Their implementation is available here:
https://github.com/larsmaaloee/
  auxiliary-deep-generative-models

The implementation here has two notable
differences from Maaloe's version.
1.  use theano but do not use lasagne
2.  maintain 5th-order tensors
These differences hopefully make the code
much easier to understand and link it closer
to the underlying mathematics. 

The implementation here has two very big
problems, however, compared to Maaloe's version.
1.  half as slow in terms of processing speed
2.  not as consistent or powerful

In the folder given_files, there are two
pdf files, both showing a handful of
test-data accuracies against training epoch.
After 200 epochs, Maaloe's version 
always gets at least 95.5% accuracy and
goes as high as 97.9%.  By contrast, my version
gets to a maximum of 96.6% but frequently 
fails to go over 95%.

I am making the code public because I need help
figuring out why it is slower and why it
does not perform as well as Maaloe's 
implementation.



In order to run the code, you first need to place
mnist.pkl.gz into the folder given_files.
Then, simply execute ``python sdgm.py'' in the
code folder to train a variational model
(SDGM by default).  

All of the code is designed for python2.
