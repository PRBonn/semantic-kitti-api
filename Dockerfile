# This file is covered by the LICENSE file in the root of this project.

# Use an official ubuntu runtime as a parent image
FROM ubuntu:16.04

# Install all system pre-reqs
# common pre-reqs
RUN apt update
RUN apt upgrade -y
RUN apt install apt-utils \
                build-essential \
                curl \
                git \
                cmake \
                unzip \
                autoconf \
                autogen \
                libtool \
                mlocate \
                zlib1g-dev \
                python \
                python3-dev \
                python3-pip \
                python3-wheel \
                python3-tk \
                wget \
                libpng-dev \
                libfreetype6-dev \
                vim \
                meld \
                sudo \
                libav-tools \
                python3-pyqt5.qtopengl \
                x11-apps \
                -y
RUN updatedb

# # Install any python pre-reqs from requirements.txt
RUN pip3 install -U pip
RUN pip3 install  scipy==0.19.1 \
                  numpy==1.14.0 \
                  torch==0.4.1 \
                  opencv_python==3.4.0.12 \
                  vispy==0.5.3 \
                  tensorflow==1.11.0 \
                  PyYAML==3.13  \
                  enum34==1.1.6 \
                  matplotlib==3.0.3

ENV PYTHONPATH /home/developer/api

# graphical interface stuff

# uid and gid
ARG uid=1000
ARG gid=1000

# echo to make sure that they are the ones from my setup
RUN echo "$uid:$gid"

# Graphical interface stuff
RUN mkdir -p /home/developer && \
    cp /etc/skel/.bashrc /home/developer/.bashrc && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer

# opengl things
ENV DEBIAN_FRONTEND "noninteractive"
# Install all needed deps
RUN apt install -y xvfb pkg-config \
                  llvm-3.9-dev \
                  xorg-server-source \
                  python-dev \
                  x11proto-gl-dev \
                  libxext-dev \
                  libx11-xcb-dev \
                  libxcb-dri2-0-dev \
                  libxcb-xfixes0-dev \
                  libdrm-dev \
                  libx11-dev;

# compile the mesa llvmpipe driver from source.
RUN mkdir -p /var/tmp/build; \
    cd /var/tmp/build; \
    wget "https://mesa.freedesktop.org/archive/mesa-18.0.1.tar.gz"; \
    tar xfv mesa-18.0.1.tar.gz; \
    rm mesa-18.0.1.tar.gz; \
    cd mesa-18.0.1; \
    ./configure --enable-glx=gallium-xlib --with-gallium-drivers=swrast,swr --disable-dri --disable-gbm --disable-egl --enable-gallium-osmesa --enable-llvm --prefix=/usr/local/ --with-llvm-prefix=/usr/lib/llvm-3.9/; \
    make -j3; \
    make install; \
    cd .. ; \
    rm -rf mesa-18.0.1;

# install mesa stuff for testing
RUN sudo apt install -y glew-utils libglew-dev freeglut3-dev \
    && wget "ftp://ftp.freedesktop.org/pub/mesa/demos/mesa-demos-8.4.0.tar.gz" \
    && tar xfv mesa-demos-8.4.0.tar.gz \
    && rm mesa-demos-8.4.0.tar.gz \
    && cd mesa-demos-8.4.0 \
    && ./configure --prefix=/usr/local \
    && make -j3 \
    && make install \
    && cd .. \
    && rm -rf mesa-demos-8.4.0

# clean the cache
RUN apt update && \
    apt autoremove --purge -y && \
    apt clean -y

ENV XVFB_WHD="1920x1080x24"\
    DISPLAY=":99" \
    LIBGL_ALWAYS_SOFTWARE="1" \
    GALLIUM_DRIVER="swr" \
    LP_NO_RAST="false" \
    LP_DEBUG="" \
    LP_PERF="" \
    LP_NUM_THREADS=""

# Set the working directory to the api location
WORKDIR /home/developer/api

# make user and home
USER developer
ENV HOME /home/developer

# Copy the current directory contents into the container at ~/api
ADD . /home/developer/api