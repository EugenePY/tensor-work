Im2Latex
==
- Recurrent Attention Model


![Example] (https://raw.githubusercontent.com/EugenePY/tensor-work/master/sandbox/exp/mnist-20160906-003950/sequence.gif)

Project Structure
--
- models : model scripts
- spaces
- costs
- training\_algorithms
- utils
- build : containing pretrain models

Dependency
--
fuel
blocks

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
[English Char Dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)
