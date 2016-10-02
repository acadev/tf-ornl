FROM dgx/keras:theano

RUN apt-get update \
 && apt-get install -y libhdf5-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install h5py

VOLUME /data
WORKDIR /data
