Im2Latex
==
- Recurrent Attention Model
- 96% test acc with simple attention.

![Example] (https://raw.githubusercontent.com/EugenePY/tensor-work/master/sandbox/exp/mnist-20160917-010151/sequence.gif)

Glimpes sensor
==
- Source image
![Example] (https://raw.githubusercontent.com/EugenePY/tensor-work/master/sandbox/model/test/cat.jpg)
- simple glimpes
![Example] (https://raw.githubusercontent.com/EugenePY/tensor-work/master/sandbox/retina0.png)
- Retina glimpes
![Example] (https://raw.githubusercontent.com/EugenePY/tensor-work/master/sandbox/retina1.png)
![Example] (https://raw.githubusercontent.com/EugenePY/tensor-work/master/sandbox/retina2.png)


Project Structure
--
- models : RAM
- costs : Expected log Likelihood, REINFORCE
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

Notes
----

Randomness for episodes
====
1. None Retina sensor(Do not have gobal image information)
  - Random initialize the location will help the network learn a meaning policy.
  - Decrease the number of the parameter of the action network first.
  - design a random scheme if there is no information then output random guess of locations.


Reference
---
- [English Char Dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)
- [Recurrent Models of Visual Attention](https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)
