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
        'MLFLOW_S3_ENDPOINT_URL': "http://<S3-IP>:<S3-PORT>",
        'MLFLOW_S3_IGNORE_TLS': "true"
    }

environment = environment | dict(settings)  

infra_k8s = KubernetesJob(
    env=environment,
    image="ghcr.io/khaosresearch/prefect-landcover:latest",
    namespace="mlops-prefect",
    image_pull_policy="Always",
    cluster_config=cluster_config_block,
    job=KubernetesJob.base_job_manifest(),
    finished_job_ttl=600
)