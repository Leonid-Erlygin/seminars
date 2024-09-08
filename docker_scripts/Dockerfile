FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

ARG DEBIAN_FRONTEND=noninteractive
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    sox \
    libsox-dev \
    libsox-fmt-all \
    build-essential \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app


# set theese arg's when using docker build
ARG GID
ARG UID
ARG NAME
USER root

RUN groupadd -g ${GID} ${NAME}
RUN adduser --disabled-password --force-badname --uid ${UID} --gid ${GID} --gecos '' --shell /bin/bash ${NAME} \
 && chown -R ${NAME}:${NAME} /app
RUN echo "${NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-${NAME}
USER ${NAME}

ENV HOME=/home/${NAME}


RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=$HOME/miniconda/bin:$PATH
COPY environment.yml /app/environment.yml
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda env update -n base -f /app/environment.yml \
 && rm /app/environment.yml \
 && conda clean -ya

