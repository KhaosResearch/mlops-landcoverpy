#!/bin/bash

CONFIG_FILE=config.conf

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found. Please create the configuration file and try again."
    exit 1
fi

# Load configuration from config.conf
source $CONFIG_FILE

# Replace placeholders with user-specific values
sed -i "s/<CLUSTER-IP>/$CLUSTER_IP/g" $(grep -rl '<CLUSTER-IP>' *)
sed -i "s/<S3-IP>/$S3_IP/g" $(grep -rl '<S3-IP>' *)
sed -i "s/<PREFECT-API-PORT>/$PREFECT_API_PORT/g" $(grep -rl '<PREFECT-API-PORT>' *)
sed -i "s/<MLFLOW-PORT>/$MLFLOW_PORT/g" $(grep -rl '<MLFLOW-PORT>' *)
sed -i "s/<S3-PORT>/$S3_PORT/g" $(grep -rl '<S3-PORT>' *)

echo "Config variables setup completed. Proceed with deployment steps of each service as outlined in the README."