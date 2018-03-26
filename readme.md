# Tensorized Deep Generative Models
## Implementations of SDGM and ADGM using Tensors
This is an implementation of a Skip Deep Generative Model (SDGM) and has flags that allow it to change to Auxiliary Deep Generative Models (ADGM), both proposed by Maaloe, Sonderby, Sonderby, and Winther.
https://arxiv.org/abs/1602.05473

Their implementation is available here:
https://github.com/larsmaaloee/auxiliary-deep-generative-models

The implementation here has two notable differences from Maaloe's version.
1.  use theano but do not use lasagne
2.  maintain 5th-order tensors

These differences hopefully make the code much easier to understand and link it closer to the underlying mathematics. 

### Update!!

The implementation here had two very big problems, however, compared to Maaloe's version.
1.  half as slow in terms of processing speed
2.  not as consistent or powerful

Both of the problems have been solved!  :)

The speed sorted itself out due to an update in theano from 0.8 to 0.9.

The consistency and power improved because of an increase in the data size.  The mnist data set originally has 50,000 training data points, 10,000 validation points, and 10,000 testing points.  We used to ignore the validation set entirely, but now we add the validation set to the training set's unlabeled data for a total of 60,000 unlabeled images.  This seems to be sufficient for squeezing out an extra 1% to 2% maximum as well as improved consistency.

###### /Update

In the folder given_files, there are two pdf files, both showing a handful of test-data accuracies against training epoch. After 200 epochs, Maaloe's version always gets at least 95.5% accuracy and goes as high as 97.9%.  With the above updates, our version also always reaches at least 95.5% and has a maximum of 97.8%.  

In order to run the code, you first need to place mnist.pkl.gz into the folder given_files. Then, simply execute ``python3 sdgm.py'' in the code folder to train a variational model (SDGM by default).  

All of the code is designed for python3 and theano 0.9.  
