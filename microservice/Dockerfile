# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:latest-gpu
RUN DEBIAN_FRONTEND=noninteractive apt-get update -yy \
    && apt-get install -yy --no-install-recommends ffmpeg libsm6 libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /aemotrics
RUN pip3 --no-cache-dir install -U pip
COPY "wheel/*" wheel/
RUN pip3 --no-cache-dir install wheel/* flask gunicorn
COPY aemotrics_service.py /aemotrics
ADD Aemotrics_V3-Nate-2021-12-20_pruned.tar.gz /aemotrics
RUN mkdir /aemotrics/videos
CMD python3 -m gunicorn aemotrics_service:app -b 0.0.0.0:8080
EXPOSE 8080
