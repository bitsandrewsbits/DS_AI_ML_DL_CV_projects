#!/bin/bash

#1 create dir for datasets
mkdir data

#2 download datasets archive
curl -L -o data/movie-critic-and-user-reviews.zip\
  https://www.kaggle.com/api/v1/datasets/download/bwandowando/rotten-tomatoes-9800-movie-critic-and-user-reviews

#3 unzip archive to data dir
unzip -j data/movie-critic-and-user-reviews.zip -d data
