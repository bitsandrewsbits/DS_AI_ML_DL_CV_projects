# syntax=docker/dockerfile:1

FROM ghcr.io/mlflow/mlflow

RUN mkdir /mlflow
RUN pip install psycopg2-binary