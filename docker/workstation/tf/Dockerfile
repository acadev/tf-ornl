#FROM nvdl.githost.io:4678/dgx/tensorflow:0.10-devel
FROM gcr.io/tensorflow/tensorflow:0.10.0-devel-gpu

# --RUN pip install keras
# --ENV KERAS_BACKEND=tensorflow
# --
# --RUN apt-get update && apt-get install -y \
# --        libzmq3-dev \
# --        python-dev \
# --        python-matplotlib \
# --        python-pandas \
# --        python-pip \
# --        python-sklearn && \
# --    rm -rf /var/lib/apt/lists/*
# --
# --RUN pip install \
# --        ipykernel \
# --        jupyter && \
# --    python -m ipykernel.kernelspec

COPY jupyter_notebook_config.py /root/.jupyter/
COPY jupyter.sh /usr/local/bin

EXPOSE 8888

WORKDIR /data
VOLUME /data
CMD jupyter-notebook
