# MLOps for Landcoverpy

## Overview

This repository hosts the essential resources for deploying the [landcoverpy](https://github.com/KhaosResearch/landcoverpy) model within our specially developed [MLOps infrastructure](https://github.com/KhaosResearch/mlops-infra/). It is designed to work seamlessly with a Kubernetes cluster, assuming that the MLOps infrastructure is already deployed. For detailed guidance on deploying each component, refer to [our mlops-infra repository](https://github.com/KhaosResearch/mlops-infra/).

## Infrastructure Components

The project relies on the following infrastructure components:

- **MLflow**: Manages the model registry.
- **PostgreSQL**: Provides the database backend required by MLflow for metadata storage.
- **Seldon Core**: Handles model deployment.
- **Prefect**: Manages workflow and pipeline orchestration.
- **MinIO**: Offers object storage solutions.
- **Prometheus**: Enables runtime monitoring and alerting.

## Repository Structure

The repository is organized into four primary directories:

### `training`
Contains all files necessary to deploy the training pipeline using Prefect. This pipeline manages the (re)training of models with Satellite imagery and validated locations, and stores the model versions in MLflow.

### `deployment`
Includes resources for deploying trained models from MLflow. It facilitates the deployment of models to a production environment.

### `prediction`
Provides a test script to verify the functionality of deployed models. The script generates randomized, valid polygons and sends them to the deployed model instances for predictions.

### `.github/workflows`
Automatically updates Docker images in the Google Container Registry (GCR) upon new commits.
