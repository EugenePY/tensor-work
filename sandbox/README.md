Im2Latex
==
- Recurrent Attention Model

![Example] (https://raw.githubusercontent.com/EugenePY/tensor-work/master/sandbox/exp/mnist-20160906-185501/sequence.gif)

Project Structure
--
- models : RAM
- costs : Expected log Likelihood
- training\_algorithms: REINFORCE algorithm
- utils
- build : containing pretrain models

Dependency
--
- Theano
- Fuel
- Blocks & Block extra
- Scikit-Image

Preparae the debug dataset
--
```shell
$ fuel-download mnist
$ fuel-convert mnist
$ echo export FUEL_DATA_PATH="/first/path/to/my/data:/second/path/to/my/data"
```

Script
--
train-exp.py

usuage:
```shell
	
python train-exp.py
```


Reference
---
- [English Char Dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)
- [Recurrent Models of Visual Attention](https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)
