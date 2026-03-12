#!/bin/bash
container_name="embed_gen_time_estimation"

sudo docker start $container_name 2> /dev/null
sudo docker run --name $container_name -p 11434:11434 -d ollama/ollama 2> /dev/null
