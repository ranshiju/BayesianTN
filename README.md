# BayesianTN
Bayesian tensor network
(see https://arxiv.org/abs/1912.12923)

To use this code, you may need to install: numpy, torch, termcolor, open-cv, matplotlib, and etc.

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

All default parameters can be found in function "parameter_tn_bp_mnist" in "TNNalgorithms.py"

## Layers of BTN

All layers can be mixedly used with the layers in torch.nn.

The modules for the layers in BTN can be found in "TNNalgorithms.py", particularly the class "TTN_Pool_2by2to1".

You may find many other kinds of layers in the file, which were not mentioned in the arXiv paper. They are for testing ideas.


## The "TTN_Pool_2by2to1" layer

This layer maps the probability distributions of (Lx * Ly) sets to that of (Lx/2 * Ly/2) sets. If Lx or Ly is odd, the last row or column will be copied to make it even.

### Input:

* input.shape = (num_samples, channels, sx, sy), if the ndimension of the input is 4

* input.shape = (num_samples, channels, d, sx, sy), if the ndimension of the input is 5

* Note:  
  num_samples: number of samples  
  channels: number of channels  
  sx: number of rows of the sets (data)  
  sy: number of columns of the sets (data)  
  d: dimension of each set

### Parameters:

* c_in: the number of input channels; int

* c_out: the number of output channels; int

* nx: the number of rows of the tensors in the layer; int

* ny: the number of columns of the tensors in the layer; int

* din: the dimension of the in-going indexes; int

* dout: the dimension of the out-going indexes; int

* device: cpu or cuda; default is to priority choose cuda (GPU) if available

* dtype: data type; default is torch.float32

* ini_way: the way to initialize the tensors

* out_dims: the ndimension of the output data

* simple_chl: whether to use simple way to deal with the channel. If simple_chl==True, one should take c_in=c_out

* if_pre_proc_T: if pre-process the tensors

* add_bias: if add bias

### To realize the simple tree BTN in the arXiv paper, we take:

* c_in = c_out = 1

* ini_way = 'randn'

* out_dims = 5

* if_pre_proc_T = 'square'  (to realize the rotation optimization proposed in the arXiv)

* add_bias = False

More instructions are to be added :)

### As you can see, I am a researcher in quantum physics, and not so good at programming. Therefore, the code is a little bit chaotic. You are welcome and appreciated to modify and/or use this code. If you think this code helps, please kindly cite the arXiv paper. Thanks a lot for your interest.


Cheers,

Shi-Ju Ran

sjran@cnu.edu.cn  
Associate professor  
Department of Physics  
Capital Normal University  
Beijing, 100048 China
