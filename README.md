# MLOps for Landcoverpy

This repository contains resources necessary for deploying the [landcoverpy](https://github.com/KhaosResearch/landcoverpy) model within the [developed MLOps infrastructure](https://github.com/KhaosResearch/mlops-infra/).

## Structure

### `deployment`
This directory holds the resources required to containerize an instance of landcoverpy using a Seldon wrapper. To deploy one or more instances of the model, configure the environment variables in `deployment/deploy_landcover.yaml` and run `kubectl apply -f deploy_landcover.py`.

### `prediction`
Here you will find a test script for verifying the functionality of the deployed model. This script generates randomized polygons that meet specific criteria to ensure their validity and sends them to the selected model instances.

### `training`
Contains all necessary files to deploy the training pipeline on the Prefect instance set up within our infrastructure.
