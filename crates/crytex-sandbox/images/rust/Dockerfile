FROM rust:1.86-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10000 crytex && useradd -m -u 10000 -g crytex -s /bin/bash crytex

WORKDIR /workspace
USER crytex

# Keep the pre-installed rustup toolchain (read-only) but put cargo caches on
# the writable /tmp tmpfs that DockerBackend mounts at runtime.
ENV CARGO_HOME=/tmp/cargo
ENV PATH="/usr/local/cargo/bin:${PATH}"

CMD ["cargo", "--version"]
