FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10000 crytex && useradd -m -u 10000 -g crytex -s /bin/bash crytex

WORKDIR /workspace
USER crytex

# Writable cache path when the rootfs is read-only.
ENV PIP_CACHE_DIR=/tmp/pip-cache
ENV PYTHONUNBUFFERED=1

CMD ["python", "--version"]
