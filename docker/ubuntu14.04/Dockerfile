# install the tlpipe

FROM ubuntu:14.04
MAINTAINER Shifan Zuo "sfzuo@bao.ac.cn"
ENV REFRESHED_AT 2017-08-02

# update package info
RUN apt-get -yqq update

# install prerequisites for Python
RUN apt-get -yqq install python-dev

# install python-pip
RUN apt-get -yqq install python-pip

# upgrade pip to the latest version
RUN pip install -U pip

# install numpy, scipy, matplotlib
RUN pip install numpy scipy matplotlib
# RUN apt-get -yqq install python-numpy python-scipy python-matplotlib

# install MPI environment, either mpich or openmpi
# install mpich
RUN apt-get -yqq install mpich

# install openmpi
# RUN apt-get -yqq install openmpi-bin libopenmpi-dev

# install mpi4py
RUN pip install mpi4py

# install HDF5
RUN apt-get -yqq install libhdf5-dev

# install h5py
RUN pip install h5py

# install pyephem
RUN pip install pyephem

# install healpy
RUN pip install healpy

# install cython
RUN pip install cython

# install git
RUN apt-get -yqq install git

# install cora
# RUN pip install git+https://github.com/zuoshifan/cora.git
RUN pip install git+https://github.com/radiocosmology/cora.git

# install aipy
RUN pip install git+https://github.com/zuoshifan/aipy.git@zuo/develop

# install caput
RUN pip install git+https://github.com/zuoshifan/caput.git@zuo/develop

# install tlpipe
# RUN pip install git+https://github.com/TianlaiProject/tlpipe.git@zuo/develop
WORKDIR /usr/src
RUN git clone https://github.com/TianlaiProject/tlpipe.git
WORKDIR /usr/src/tlpipe
# RUN git checkout zuo/develop
Run python setup.py develop

# clean downloaded packages
WORKDIR /
RUN apt-get autoremove
