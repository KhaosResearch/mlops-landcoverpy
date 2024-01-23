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

### `gui`
Provides a web-based user interface for interacting with the deployed models. It allows users to interact with a map and draw polygons to generate, visualise and download predictions.

### `.github/workflows`
Automatically updates Docker images in the Google Container Registry (GCR) upon new commits.

## Before you begin

Before starting to integrate the models in the infrastructure, please perform the following steps:

1. Clone this repository:

  ```bash
    git clone https://github.com/KhaosResearch/mlops-landcoverpy.git
    cd mlops-landcoverpy
  ```

2. Modify the `config.conf` file with the infrastructure service addresses, ports, etc.

3. Execute the `setup.sh` script to replace the placeholders in all the files with the values in `config.conf`:

  ```bash
    chmod +x setup.sh
    ./setup.sh
  ```

## How to integrate landcoverpy models in the infrastructure

The following sections describe how to integrate landcoverpy models in the infrastructure.

### Deploy the base model in MLflow


1. Move to the `training` directory:

  ```bash
    cd training
  ```
2. Train one instance of the model using the [landcoverpy](https://github.com/KhaosResearch/landcoverpy) repository. This will create several files such as the model instance `model.joblib`, model's metadata `metadata.json`, and a confusion matrix `confusion_matrix.png` (also `confusion_matrix.csv`). Move these files, along with the train and test data used to a `base_model` directory.
3. Execute the `upload_base_model.py` script to upload the first version of the model to the model registry. Having the base model, all the subsequent versions will be trained and uploaded automatically using Prefect.
   ```bash
    python upload_base_model.py
   ```
4. Deploy the base version of the model to the production environment using the `deployment/deploy_landcover.yaml` file. You will need to use the same environment variables used for training the base model.
   ```bash
    cd ../deployment
    kubectl apply -f deploy_landcover.yaml
   ```

### Deploy the training pipeline

1. Move to the `training` directory:

  ```bash
    cd training
  ```
2. To let prefect uploading deployment files to the storage block (S3), environment variable `FSSPEC_S3_ENDPOINT_URL` has to be set.

  ```bash
    export FSSPEC_S3_ENDPOINT_URL=http://<S3-IP>:<S3-PORT>
  ```

3. Create a deployment for the training pipeline providing the K8s infrastructure. 

  ```bash
    prefect deployment build -n retraining-flow-deployment-k8s \
    -p k8s-pool -ib kubernetes-job/k8s-infra -sb s3/khaos-minio \
    -o retraining_flow_deployment.yaml \
    pipeline_retrain.py:retraining_flow
  ```
4. Upload to prefect the deployment for the training pipeline linked to the infrastructure.
 
  ```bash
    prefect deployment apply test_flow_deployment.yaml
  ```

5. Deploy to the k8s cluster agents able to run the retraining pipeline.

  ```bash
    kubectl apply -f prefect_agent_deployment.yaml
  ```

Now it is possible to retrain the model automatically from Prefect UI or using the CLI indicating new data location in the object storage.

## Making predictions to the models

Once models are deployed, a REST API and a GRPC API are available through the istio gateway. The easiest way to make predictions to the models is using the GUI. However, it is also possible to make predictions using the CLI or the Python SDK.

Example Python code to make predictions using the Python SDK:

```py
    sc = SeldonClient(
        deployment_name=f"landcover-seldon",
        namespace="mlops-seldon"
    )
    res = sc.predict(
        transport="grpc",
        gateway="istio",
        gateway_endpoint=gateway_endpoint,
        raw_data={"strData":json.dumps(geojson)}
    )
    print(res.response["jsonData"]["result"])
```

### Using the GUI

To locally run the GUI, you will need to install the dependencies in the `gui` directory:

```bash
    cd gui
    pip install -r requirements.txt
```

Then, you can run the GUI using the following command:

```bash
    streamlit run streamlit_app.py
```

The command will start a local server and open a browser window with the GUI. The GUI allows users to interact with a map and draw polygons to generate, visualise and download predictions.