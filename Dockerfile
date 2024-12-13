FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# INSTALL_TVM also enables / disables LLVM installation
# INSTALL_LLVM also enables / disables model-opt installation
# For TORCH_CUDA_ARCH_LIST see https://en.wikipedia.org/wiki/CUDA#GPUs_supported (You only need to specify the compute capability of the GPU / CUDA version you are using)

ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.2 7.5 8.0 8.6 8.7 9.0+PTX" 
ARG INSTALL_TVM=true
ARG INSTALL_TENSORRT=true 

ARG TENSORRT_URL=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.4.0/tars/TensorRT-10.4.0.26.Linux.x86_64-gnu.cuda-12.6.tar.gz

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
    jq \
    gnupg \
    lsb-release

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


RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.2/cmake-3.31.2.tar.gz && \
    tar -zxvf cmake-3.31.2.tar.gz && \
    cd cmake-3.31.2 && \
    ./bootstrap && \
    make -j8 && \
    make install

RUN apt-get clean -y && \
    rm -rf /build/* && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install uv



RUN if [ "$INSTALL_TVM" = "true" ]; then \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libzstd-dev && \
    wget https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh && \
    chmod +x /tmp/llvm.sh && \
    /tmp/llvm.sh 18 && \
    rm -rf /tmp/llvm.sh && \
    apt-get install -y --no-install-recommends \
    llvm-18 llvm-18-dev llvm-18-tools libpolly-18-dev clang-18 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/llvm-config-18 /usr/bin/llvm-config && \
    ln -s /usr/bin/clang-18 /usr/bin/clang; \
fi

# apache-tvm
# https://tvm.apache.org/docs/install/from_source.html#step-1-install-dependencies 
RUN if [ "$INSTALL_TVM" = "true" ]; then \
    mkdir -p /build/tvm && \
    git clone --recursive https://github.com/apache/tvm && \
    cd tvm && \
    echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake && \
    echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake && \
    echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake && \
    echo "set(USE_CUDA   ON)" >> config.cmake && \
    echo "set(USE_METAL  OFF)" >> config.cmake && \
    echo "set(USE_VULKAN OFF)" >> config.cmake && \
    echo "set(USE_OPENCL ON)" >> config.cmake && \
    echo "set(USE_CUBLAS ON)" >> config.cmake && \
    echo "set(USE_CUDNN  ON)" >> config.cmake && \
    echo "set(USE_CUTLASS OFF)" >> config.cmake && \
    cmake ./ && \
    cd /build/tvm && \
    cmake --build . --parallel 16 && \
    ln -s /build/tvm/libtvm.so /usr/bin/libtvm.so && \
    ln -s /build/tvm/libtvm_runtime.so /usr/bin/libtvm_runtime.so && \
    ln -s /build/tvm/libtvm_allvisible.so /usr/bin/libtvm_allvisible.so && \
    cd /build/tvm && \
    pip install -e python; \
fi

# Install TensorRT dev environment
# https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/docker/Dockerfile
RUN if [ "$INSTALL_TENSORRT" = "true" ]; then \
    mkdir -p /build/tensorrt && \
    cd /build/tensorrt && \
    wget -q -O tensorrt.tar.gz $TENSORRT_URL && \
    tar -xf tensorrt.tar.gz && \
    cp TensorRT-*/bin/trtexec /usr/local/bin && \
    cp TensorRT-*/include/* /usr/include/x86_64-linux-gnu && \
    python -m pip install TensorRT-*/python/tensorrt-*-cp310-none-linux_x86_64.whl && \
    mkdir -p /usr/local/lib/python3.10/dist-packages/tensorrt_libs && \
    cp -a TensorRT-*/targets/x86_64-linux-gnu/lib/* /usr/local/lib/python3.10/dist-packages/tensorrt_libs && \
    rm -rf TensorRT-*.Linux.x86_64-gnu.cuda-*.tar.gz TensorRT-* tensorrt.tar.gz; \
fi

ENV TRT_LIB_PATH=/usr/local/lib/python3.10/dist-packages/tensorrt_libs
ENV LD_LIBRARY_PATH=$TRT_LIB_PATH:$LD_LIBRARY_PATH

RUN if [ "$INSTALL_TENSORRT" = "true" ]; then \
uv pip install --system "nvidia-modelopt[torch,onnx]" -U && \
    python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"; \
fi


RUN mkdir -p /build/tensorrt && \
    cd /build/tensorrt && \ 
    wget -q -O tensorrt.tar.gz $TENSORRT_URL && \
    tar -xf tensorrt.tar.gz && \
    cp TensorRT-*/bin/trtexec /usr/local/bin && \
    cp TensorRT-*/include/* /usr/include/x86_64-linux-gnu && \
    python -m pip install TensorRT-*/python/tensorrt-*-cp310-none-linux_x86_64.whl && \ 
    cd /build/tensorrt && \
    mkdir -p /usr/local/lib/python3.10/dist-packages/tensorrt_libs && \
    cp -a TensorRT-*/targets/x86_64-linux-gnu/lib/* /usr/local/lib/python3.10/dist-packages/tensorrt_libs && \
    rm -rf TensorRT-*.Linux.x86_64-gnu.cuda-*.tar.gz TensorRT-* tensorrt.tar.gz
ENV TRT_LIB_PATH=/usr/local/lib/python3.10/dist-packages/tensorrt_libs
ENV LD_LIBRARY_PATH=$TRT_LIB_PATH:$LD_LIBRARY_PATH


ADD . /build/alma

RUN cp /build/alma/requirements.txt /build/alma/requirements.modified.txt && \
    if [ "$INSTALL_TENSORRT" != "true" ]; then \
        sed -i '/torch-tensorrt/d' /build/alma/requirements.modified.txt; \
    fi && \
    if [ "$INSTALL_TVM" != "true" ]; then \
        sed -i '/apache-tvm/d' /build/alma/requirements.modified.txt; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    zip \
    g++ \
    openjdk-11-jdk \
    ca-certificates

RUN curl -sSL "https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64" \
    -o /usr/local/bin/bazelisk && \
    chmod +x /usr/local/bin/bazelisk && \
    ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel && \
    bazel

RUN mkdir -p /build/openxla && \
    cd /build && \
    git clone https://github.com/openxla/openxla.git && \
    cd /build/openxla && \
    ./configure.py  --backend=CUDA --backend=CPU && \
    bazel build //:all && \
    bazel clean --expunge

RUN uv pip install --system -r /build/alma/requirements.modified.txt

RUN cd /build/alma && \
    pip install -e ./
