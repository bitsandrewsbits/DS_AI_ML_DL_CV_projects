# syntax=docker/dockerfile:1

FROM jupyter/minimal-notebook

ADD requirements.txt /

RUN pip install -r /requirements.txt