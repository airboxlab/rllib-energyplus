FROM ubuntu:22.04

LABEL org.opencontainers.image.source="https://github.com/airboxlab/rllib-energyplus"
LABEL org.opencontainers.image.description="EnergyPlus Env for Reinforcement Learning in Building Energy Management Systems"

ARG EPLUS_VERSION=23-2-0
ARG EPLUS_DL_URL=https://github.com/NREL/EnergyPlus/releases/download/v23.2.0/EnergyPlus-23.2.0-7636e6b3e9-Linux-Ubuntu22.04-x86_64.sh

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/ray \
    TZ=UTC

USER root
SHELL ["/bin/bash", "-c"]

COPY requirements.txt /root/rllib-energyplus/

RUN \
    # install E+
    apt-get update -qq && apt-get install -y wget git && \
    cd /tmp && \
    wget --quiet "${EPLUS_DL_URL}" && \
    export eplus_install="$(echo "${EPLUS_DL_URL}" | rev | cut -d'/' -f1 | rev)" && \
    (echo "y"; echo ""; echo "y";) | bash "$eplus_install" && \
    rm "$eplus_install" && \
    # install python3.10
    apt-get install -y python3.10 python3.10-dev python3.10-distutils && \
    # install pip
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py && \
    # install requirements
    cd /root/rllib-energyplus && \
    # since pip implicitly uses --require-hashes if a hash is present, but
    # git repos don't have hashes, so we need to split the requirements file
    grep 'git\+' requirements.txt > requirements-vcs.txt && \
    grep 'git\+' -v requirements.txt > requirements-hashed.txt && \
    pip install --no-cache-dir -r requirements-vcs.txt && \
    pip install --no-cache-dir -r requirements-hashed.txt && \
    # cleanup
    apt autoremove -qq -y && apt-get clean -qq && rm -rf /var/lib/apt/lists/*

COPY rleplus /root/rllib-energyplus/rleplus

ENV PYTHONPATH=/root/rllib-energyplus

ENTRYPOINT "/bin/bash"
