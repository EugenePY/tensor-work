Im2Latex
--
This project using some desing pattern borrowing from PyLearn2

Project Structure
--
models : model scripts
spaces
costs
training\_algorithms
utils
build : containing pretrain models

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
train_exp.py

usuage:
```shell
	
python train_exp.py
```


Reference
---
[English Char Dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)
