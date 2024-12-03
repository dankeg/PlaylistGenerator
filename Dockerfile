# Use amazonlinux as base image
FROM --platform=amd64 amazonlinux:latest AS base

# Install system dependencies
RUN dnf update -y --releasever=latest \
    && dnf upgrade -y --releasever=latest \
    && dnf clean all \
    && dnf install gcc python3-devel pkgconf -y

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set environment variables
ENV PATH="/root/.local/bin:$PATH"

# ---------------------------------------------------------------------
FROM base AS poetry

# Confirm Poetry installation
RUN poetry --version

WORKDIR /opt/pipeline

# Copy the entire project directory (including source code)
COPY . /opt/pipeline

# Install pip (for compatibility)
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python3 -

# Build the package and install it with pip
RUN poetry build && pip install dist/*.whl

# ---------------------------------------------------------------------
FROM poetry AS prod

WORKDIR /opt/pipeline

# Install additional runtime dependencies
RUN pip install dm-reverb[tensorflow]
RUN pip install "tensorflow-cpu==2.15.1"

# Expose Streamlit port
EXPOSE 8501

# # Set the entrypoint
# ENTRYPOINT ["streamlit", "run", "--server.address=0.0.0.0"]
