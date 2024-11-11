ARG DEBIAN_FRONTEND=noninteractive
FROM ubuntu:22.04

WORKDIR /app

# Update the system and install dependencies
RUN apt-get update && apt-get install -y software-properties-common

# Add the deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa

# Update the system again and install Python 3.8
RUN apt-get update -y
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y python3.8 python3.8-distutils python3.8-venv python3.8-dev wget \
    python3-tk

RUN wget https://bootstrap.pypa.io/get-pip.py
# Install pip
RUN python3.8 get-pip.py
# Verify the installation
RUN python3.8 --version

RUN echo "## Update and install packages"

RUN echo "### Support packages:"

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update -qq -y --fix-missing \
  && apt-get install -qq -y gcc g++ curl make xxd wget apt-utils dos2unix software-properties-common git autoconf \
    automake libtool unzip cmake build-essential protobuf-compiler libprotoc-dev graphviz graphviz-dev locales locales-all sudo > /dev/null


COPY requirements.txt /app/requirements.txt
COPY MANIFEST.in /app/MANIFEST.in
COPY setup.py /app/setup.py
COPY conversion_toolchain /app/conversion_toolchain
COPY scripts/ /app/scripts

RUN echo "### Installing Python dependencies:"
RUN python3.8 -m venv /app/env
ENV PATH="/app/env/bin:$PATH"
RUN pip install -r requirements.txt

ENTRYPOINT ["bash", "/app/scripts/convert.sh"]