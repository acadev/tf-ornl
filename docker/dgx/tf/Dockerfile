FROM compute.nvidia.com/nvidia/tensorflow

ENV NCCL_TOPOLOGY=CUBEMESH

RUN pip install keras
ENV KERAS_BACKEND=tensorflow

RUN apt-get update && apt-get install -y \
        libzmq3-dev \
        python-dev \
        python-matplotlib \
        python-pandas \
        python-pip \
        python-sklearn && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
        ipykernel \
        jupyter && \
    python -m ipykernel.kernelspec

COPY jupyter_notebook_config.py /root/.jupyter/
COPY jupyter.sh /usr/local/bin

EXPOSE 8888

WORKDIR /data
VOLUME /data
CMD jupyter.sh
