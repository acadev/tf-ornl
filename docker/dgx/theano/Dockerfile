FROM dgx/cudnn5

RUN apt-get update \
 && apt-get install -y python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install Theano

ENV THEANO_FLAGS="mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1"

