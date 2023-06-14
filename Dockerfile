FROM ubuntu:22.04

LABEL org.opencontainers.image.source="https://github.com/airboxlab/rllib-energyplus"

ARG EPLUS_VERSION=23-1-0
ARG EPLUS_DL_URL=https://github.com/NREL/EnergyPlus/releases/download/v23.1.0/EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/ray \
    TZ=UTC

USER root
SHELL ["/bin/bash", "-c"]

COPY requirements.txt /root/rllib-energyplus/

RUN \
    # install E+
    apt-get update -qq && apt-get install -y wget && \
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
    pip install --no-cache-dir -r requirements.txt && \
    # cleanup
    apt autoremove -qq -y && apt-get clean -qq && rm -rf /var/lib/apt/lists/*

COPY model.idf LUX_LU_Luxembourg.AP.065900_TMYx.2004-2018.epw run.py /root/rllib-energyplus/

ENV PYTHONPATH="/usr/local/EnergyPlus-${EPLUS_VERSION}"
ENTRYPOINT "/bin/bash"
