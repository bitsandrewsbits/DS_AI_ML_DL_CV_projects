#!/bin/bash

# make data dir
mkdir data
# download dataset from Kaggle
curl -L -o data/global-soil-characteristics-dataset-1-million.zip\
	https://www.kaggle.com/api/v1/datasets/download/hossam82/global-soil-characteristics-dataset-1-million

# unzip dataset archive to data dir
unzip data/global-soil-characteristics-dataset-1-million.zip -d data
