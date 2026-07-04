FROM node:22-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10000 crytex && useradd -m -u 10000 -g crytex -s /bin/bash crytex

WORKDIR /workspace
USER crytex

# Writable cache path when the rootfs is read-only.
ENV NPM_CONFIG_CACHE=/tmp/npm
ENV PATH="/workspace/node_modules/.bin:${PATH}"

CMD ["node", "--version"]
