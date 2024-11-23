FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

WORKDIR /build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    software-properties-common \
    build-essential \
    tzdata \
    git wget \
    zlib1g-dev libomp-dev\
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
    libunistring-dev libopencv-dev python3-opencv \
    tmux

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

# Install other packages
RUN uv pip install --system \
    numpy==1.26.4 \
    torch==2.4.0 \
    torchvision==0.19.0 \
    black==24.4.1 \
    isort==5.13.2 \
    tqdm==4.66.1 \
    ipdb \
    pre-commit \
    ruff \
    pylint \
    mkl \
    mkl-include \
    cmake \
    cffi \
    Cython \
    matplotlib \
    pip-system-certs \
    cupy-cuda12x \
    pybind11[global] \
    typing_extensions