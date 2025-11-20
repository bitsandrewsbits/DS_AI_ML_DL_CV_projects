#!/bin/bash

SUBNET="173.18.0.0/16"
GATEWAY="173.18.0.1"

JupyterNotebook="173.18.0.2"
MLFlow_Server_IP="173.18.0.3"
PostgreSQL_DB_IP="173.18.0.4"

PostgreSQL_DB_username=postgres
PostgreSQL_DB_password=postgres_password

# write to .env file
echo "PROJECT_DIR=$(pwd)" > .env
echo "SUBNET=$SUBNET" >> .env
echo "GATEWAY=$GATEWAY" >> .env
echo "JUPYTER_NOTEBOOK_IP=$JupyterNotebook" >> .env
echo "MLFLOW_IP=$MLFlow_Server_IP" >> .env
echo "POSTGRESQL_DB_IP=$PostgreSQL_DB_IP" >> .env
echo "POSTGRESQL_DB_USERNAME=$PostgreSQL_DB_username" >> .env
echo "POSTGRESQL_DB_PASSWORD=$PostgreSQL_DB_password" >> .env
echo "MLFLOW_TRACKING_URI=postgresql+psycopg2://$PostgreSQL_DB_username:$PostgreSQL_DB_password@$PostgreSQL_DB_IP:5432/postgres" >> .env