ARG DEBIAN_FRONTEND=noninteractive
FROM ubuntu:22.04

WORKDIR /app

# Python 3.10 is Ubuntu 22.04's system Python (no PPA needed). Its venv bundles pip.
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update -y && apt-get install -y \
    python3 python3-venv python3-dev python3-distutils wget python3-tk
RUN python3 --version

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
RUN python3 -m venv /app/env
ENV PATH="/app/env/bin:$PATH"
RUN pip install -r requirements.txt

ENTRYPOINT ["bash", "/app/scripts/convert.sh"]