import os

# IF you change the base image, you need to rebuild all images (run with --force_rebuild)
_DOCKERFILE_BASE = r"""
FROM buildpack-deps:jammy

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    cmake \
    libboost-all-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ca-certificates-java \
    openjdk-21-jdk \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Go with architecture detection
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        GOARCH="amd64"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        GOARCH="arm64"; \
    else \
        false; \
    fi && \
    curl -L "{go_url}" -o go.tar.gz && \
    tar -C /usr/local -xzf go.tar.gz && \
    rm go.tar.gz
ENV PATH="/usr/local/go/bin:${{PATH}}"

# Install Rust
ADD {rustup_url} /tmp/rustup.sh
RUN chmod +x /tmp/rustup.sh && /tmp/rustup.sh -y && rm /tmp/rustup.sh
ENV PATH="/root/.cargo/bin:${{PATH}}"

# Install Node.js and dependencies
RUN curl -fsSL {nodesource_url} | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /npm-install && \
    cd /npm-install && \
    npm init -y && \
    npm install \
    jest \
    @babel/core@7.25.2 \
    @exercism/babel-preset-javascript@0.2.1 \
    @exercism/eslint-config-javascript@0.6.0 \
    @types/jest@29.5.12 \
    @types/node@20.12.12 \
    babel-jest@29.6.4 \
    core-js@3.37.1 \
    eslint@8.49.0

# Download and install conda
RUN wget '{miniconda_url}' -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/miniconda3
# Add conda to PATH
ENV PATH=/opt/miniconda3/bin:$PATH
# Add conda to shell startup scripts like .bashrc (DO NOT REMOVE THIS)
RUN conda init --all
RUN conda config --append channels conda-forge

RUN adduser --disabled-password --gecos 'dog' nonroot
"""

_DOCKERFILE_ENV = r"""FROM --platform={platform} pb.base.{arch}:latest

COPY ./setup_env.sh /root/
RUN chmod +x /root/setup_env.sh
RUN /bin/bash -c "source ~/.bashrc && /root/setup_env.sh"

WORKDIR /testbed/

# Automatically activate the testbed environment
RUN echo "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed" > /root/.bashrc
"""

_DOCKERFILE_INSTANCE = r"""FROM --platform={platform} {env_image_name}

COPY ./setup_repo.sh /root/
COPY ./polyglot /polyglot
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
"""


# Installer locations for the Polyglot base image are supplied by the user.
# WGM_GO_TARBALL_URL may contain the shell variable $GOARCH (amd64/arm64) and
# WGM_MINICONDA_INSTALLER_URL the placeholder {conda_arch} (x86_64/aarch64).
BASE_IMAGE_URLS = {
    "go_url": "WGM_GO_TARBALL_URL",
    "rustup_url": "WGM_RUSTUP_INIT_URL",
    "nodesource_url": "WGM_NODESOURCE_SETUP_URL",
    "miniconda_url": "WGM_MINICONDA_INSTALLER_URL",
}


def base_image_urls(conda_arch):
    missing = [name for name in BASE_IMAGE_URLS.values() if not os.environ.get(name)]
    if missing:
        raise RuntimeError("Set the Polyglot base-image installer URLs: " + ", ".join(missing))
    values = {key: os.environ[name] for key, name in BASE_IMAGE_URLS.items()}
    values["miniconda_url"] = values["miniconda_url"].replace("{conda_arch}", conda_arch)
    return values


def get_dockerfile_base(platform, arch):
    if arch == "arm64":
        conda_arch = "aarch64"
    else:
        conda_arch = arch
    return _DOCKERFILE_BASE.format(**base_image_urls(conda_arch))


def get_dockerfile_env(platform, arch):
    return _DOCKERFILE_ENV.format(platform=platform, arch=arch)


def get_dockerfile_instance(platform, env_image_name):
    return _DOCKERFILE_INSTANCE.format(platform=platform, env_image_name=env_image_name)
