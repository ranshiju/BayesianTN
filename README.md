# BayesianTN
Bayesian tensor network
(see https://arxiv.org/abs/1912.12923)

==========================================================
## How to build a Bayesian TN (BTN)?

You may find an example of a BTN in the file "BayesTN.py".

In the class "VL_Bayes_sum1_Vsoftmax_TN_BP", you can find the BTN with a simple tree structure (same as the arXiv paper) for MNIST and fashion-MNIST datasets.

I will explain some more details later below.

## How to use a Bayesian TN (BTN)?

The optimization algorithm can be found as the function "tn_multi_classifier_bp_mnist" in "TNN algorithms.py". 

You may refer to "RunTTN.py" to see how to use a built BTN.Let me explain this file in some details.

```
import CollectedTNN as ctnn
from TNNalgorithms import tn_multi_classifier_bp_mnist as mps_bp

# import the BTN
from BayesTN import Paras_VL_Bayes_sum1_Vsoftmax_TN
# Set all default parameters
para = Paras_VL_Bayes_sum1_Vsoftmax_TN()

# Change the parameters as you want
para['d'] = 2  # dimension of the root sets
para['chi'] = 2  # dimension of the hidden sets
para['dataset'] = 'fashion-mnist'  # 'mnist' or 'fashion-mnist'
para['batch_size'] = 2000  # batch size
para['lr'] = 1e-3  # learning rate

mps_bp(para)  # call the optimization algorithm
```

## Layers of BTN


More instructions are to be added in the short future :)
