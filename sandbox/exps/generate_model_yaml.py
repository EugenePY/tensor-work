import pylearn2.config.yaml_parse as yaml

fp = open('./exps/RAM.yaml')
f = yaml.load(fp)
print f
