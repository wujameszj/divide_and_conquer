FROM --platform=linux/amd64 pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
# FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:24.03-py3
# FROM --platform=linux/amd64 nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED 1
# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

COPY --chown=user:user . /ProPainter/
# COPY --chown=user:user dream/temp/728_408_100000.pth /ProPainter/model/

WORKDIR /ProPainter/

RUN python -m pip install \
    --user --no-cache-dir \
    --requirement requirements.txt

ENTRYPOINT ["python", "scripts/docker_main_oddeven.py"]
