from landcoverpy.config import settings
from prefect.blocks.kubernetes import KubernetesClusterConfig
from prefect.filesystems import S3
from prefect.infrastructure.kubernetes import KubernetesJob

cluster_config_block = KubernetesClusterConfig.load("k8s-config")
s3_block = S3.load("khaos-minio")

environment = {
        'PREFECT_API_URL': 'http://<CLUSTER-IP>:<PREFECT-API-PORT>/api',
        'FSSPEC_S3_ENDPOINT_URL': 'http://<S3-IP>:<S3-PORT>',
        'AWS_ACCESS_KEY_ID': s3_block.aws_access_key_id.get_secret_value(),
        'AWS_SECRET_ACCESS_KEY': s3_block.aws_secret_access_key.get_secret_value(),
        'MLFLOW_TRACKING_URI': 'http://<CLUSTER-IP>:<MLFLOW-PORT>',
        'MLFLOW_S3_ENDPOINT_URL': "http://<S3-IP>:<S3-PORT>",
        'MLFLOW_S3_IGNORE_TLS': "true"
    }

environment = dict(settings) | environment

customizations = [
    {
        "op": "add",
        "path": "/spec/template/spec/containers/0/resources",
        "value": {
            "requests": {
                "cpu": "2",
                "memory": "30gi"
            },
            "limits": {
                "cpu": "2",
                "memory": "30Gi"
            }
        },
    },{
        "op": "replace",
        "path": "/spec/template/spec/parallelism",
        "value": 5,
    },{
        "op": "remove",
        "path": "/spec/template/spec/completions",
    }
]

infra_k8s = KubernetesJob(
    env=environment,
    image="ghcr.io/khaosresearch/prefect-landcover:latest",
    namespace="mlops-prefect",
    image_pull_policy="Always",
    cluster_config=cluster_config_block,
    job=KubernetesJob.base_job_manifest(),
    customizations=customizations,
    finished_job_ttl=600
)

infra_k8s.save("k8s-infra-retraining", overwrite=True)