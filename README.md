# GMLS-Nets

<div  align="center">
<img src="misc/overview.png" width = "100%" />
</div>

__PyTorch implementation of GMLS-Nets.__

__Installation__

*Method 1:* 

Available from PyPi you can install as

```pip install gmlsnets-pytorch```

*Method 2:*

Download the [gmlsnets_pytorch-1.0.0.tar.gz](https://github.com/atzberg/gmls-nets-testing/blob/master/gmlsnets_pytorch-1.0.0.tar.gz) file above, then uncompress 

``tar -xvf gmlsnets_pytorch-1.0.0.tar.gz`` 

For local install, please be sure to edit in your codes the path location of base directory by adding

``sys.path.append('package-path-location-here');`` 

Note the package resides in the sub-directory ``./gmlsnets_pytorch-1.0.0/gmlsnets_pytorch/``

__Packages__ 

Please be sure to install [PyTorch](https://pytorch.org/) package >= 1.2.0 with Python 3 (ideally >= 3.7).  Also, be sure to install the following packages: numpy>=1.16, scipy>=1.3, matplotlib>=3.0.

__Use__

For examples and documentation, see

[Examples](https://github.com/atzberg/gmls-nets/tree/master/examples)

[Documentation](http://web.math.ucsb.edu/~atzberg/gmlsnets_docs/html/index.html)

__Additional Information__

If you find these codes or methods helpful for your project, please cite: 

*GMLS-Nets: A Framework for Learning from Unstructured Data,* 
N. Trask, R. G. Patel, B. J. Gross, and P. J. Atzberger, arXiv:1909.05371, (2019), [[arXiv]](https://arxiv.org/abs/1909.05371).
```
@article{trask_patel_gross_atzberger_GMLS_Nets_2019,
  title={GMLS-Nets: A framework for learning from unstructured data},
  author={Nathaniel Trask, Ravi G. Patel, Ben J. Gross, Paul J. Atzberger},
  journal={arXiv:1909.05371},  
  month={September}
  year={2019}
  url={https://arxiv.org/abs/1909.05371}
}
```
For [TensorFlow](https://www.tensorflow.org/) implementation of GMLS-Nets, see https://github.com/rgp62/gmls-nets.

----

[Examples](https://github.com/atzberg/gmls-nets/tree/master/examples) | [Documentation](http://web.math.ucsb.edu/~atzberg/gmlsnets_docs/html/index.html) | [Atzberger Homepage](http://atzberger.org/)
