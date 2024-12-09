FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

WORKDIR /build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    software-properties-common \
    build-essential \
    tzdata \
    git \
    wget \
    zlib1g-dev libomp-dev \
    libffi-dev libncurses5-dev libssl-dev libreadline-dev \
    libgdm-dev libdb4o-cil-dev libpcap-dev libsqlite3-dev \
    liblzma-dev \
    libopenblas-dev liblapack-dev \
    libblas-dev libatlas-base-dev \
    libblas3 liblapack3 vim nano \
    libbz2-dev gcc gfortran gfortran-10 mercurial \
    rsync libturbojpeg0-dev \
    autoconf \
    automake \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libsdl2-dev \
    libtool \
    libxcb-xfixes0-dev \
    ninja-build \
    pkg-config \
    texinfo \
    yasm \
    libaom-dev meson \
    nasm \
    libhdf5-serial-dev libavdevice-dev \
    libunistring-dev \
    tmux \
    curl \
    unzip \
    openmpi-bin \
    libopenmpi-dev \
    git-lfs \
    jq

RUN wget -nv --show-progress --progress=bar:force:noscroll https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tar.xz  && \
    tar -xvf Python-3.10.14.tar.xz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --enable-shared && \
    make install

# Alias python and pip
RUN rm -rf /usr/bin/python && \
    rm -rf /usr/bin/pip && \
    ln -s /usr/local/bin/python3.10 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.10 /usr/bin/pip


RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0.tar.gz && \
    tar -zxvf cmake-3.29.0.tar.gz && \
    cd cmake-3.29.0 && \
    ./bootstrap && \
    make -j8 && \
    make install

RUN apt-get clean -y && \
    rm -rf /build/* && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install uv

# Install TensorRT dev environment
# https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/docker/Dockerfile
ARG TENSORRT_URL=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.4.0/tars/TensorRT-10.4.0.26.Linux.x86_64-gnu.cuda-12.6.tar.gz
RUN mkdir -p /build/tensorrt && \
    cd /build/tensorrt && \ 
    wget -q -O tensorrt.tar.gz $TENSORRT_URL && \
    tar -xf tensorrt.tar.gz && \
    cp TensorRT-*/bin/trtexec /usr/local/bin && \
    cp TensorRT-*/include/* /usr/include/x86_64-linux-gnu && \
    python -m pip install TensorRT-*/python/tensorrt-*-cp310-none-linux_x86_64.whl 
    
    # && \
RUN cd /build/tensorrt && \
    mkdir -p /usr/local/lib/python3.10/dist-packages/tensorrt_libs && \
    cp -a TensorRT-*/targets/x86_64-linux-gnu/lib/* /usr/local/lib/python3.10/dist-packages/tensorrt_libs && \
    rm -rf TensorRT-*.Linux.x86_64-gnu.cuda-*.tar.gz TensorRT-* tensorrt.tar.gz
ENV TRT_LIB_PATH=/usr/local/lib/python3.10/dist-packages/tensorrt_libs
ENV LD_LIBRARY_PATH=$TRT_LIB_PATH:$LD_LIBRARY_PATH

# # Install modelopt with all optional dependencies and pre-compile CUDA extensions otherwise they take several minutes on every docker run
RUN uv pip install --system "nvidia-modelopt[torch,onnx]" -U
ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.2 7.5 8.0 8.6 8.7 9.0+PTX"
RUN python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"

ADD requirements.txt /build/requirements.txt
RUN uv pip install --system -r /build/requirements.txt