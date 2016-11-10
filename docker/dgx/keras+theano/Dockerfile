FROM dgx/theano

RUN apt-get update && \
    apt-get install -y python-yaml && \
    rm -rf /var/lib/apt/lists/*

RUN pip install keras
ENV KERAS_BACKEND=theano

# project dependencies
RUN pip install h5py

VOLUME /data
WORKDIR /data
