# Dockerfile for linter pipeline, not for JEPA training execution!

FROM python:3.14-alpine3.23

RUN apk-get update && apk add --no-cache \
    git \
    bash \
    curl

RUN pip install --no-cache-dir ruff

WORKDIR /workdir

CMD ["bash"]