
.PHONY cudnn5 theano tensorflow keras-theano

all: ornl-tf ornl-theano

cudnn5:
	cd cudnn5 && docker build -t dgx/cudnn5 .

theano: cudnn5
	cd theano && docker build -t dgx/theano .

tensorflow:
	cd tf && docker build -t ornl/tf .

keras-theano: theano
	cd keras+theano && docker build -t ornl/theano .
