# Use amazonlinux as base image
FROM --platform=amd64 amazonlinux:latest AS base

# Install system dependencies
RUN dnf update -y --releasever=latest\
    && dnf upgrade -y --releasever=latest && dnf clean all && dnf install gcc python3-devel pkgconf -y

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set environment variables
ENV PATH="/root/.local/bin:$PATH"

# ---------------------------------------------------------------------
FROM base AS poetry

# Confirm Poetry installation
RUN poetry --version

WORKDIR /opt/pipeline

# Copy only pyproject.toml and poetry.lock to the container
COPY . /opt/pipeline

RUN poetry lock 

# RUN python3 -m venv env && source env/bin/activate
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python3 -

# Install dependencies with Poetry
RUN poetry build && pip install dist/*.whl

# ---------------------------------------------------------------------
FROM poetry AS prod

RUN pip install dm-reverb[tensorflow]