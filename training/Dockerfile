FROM prefecthq/prefect:2.10.4-python3.11-kubernetes

LABEL org.opencontainers.image.source=https://github.com/KhaosResearch/mlops-landcoverpy/
LABEL org.opencontainers.image.description="Prefect retraining flow image"
LABEL org.opencontainers.image.licenses=MIT

WORKDIR /home
RUN mkdir -p -v /home/files/tmp
RUN chown -R 8888 /home
RUN chmod -R 755 /home

COPY ./requirements.txt /home
COPY ./files/deploy_landcover.yaml /home/files

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Needed ENV to define outside:
# PREFECT_API_URL, FSSPEC_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, MLFLOW_S3_ENDPOINT_URL, MLFLOW_S3_IGNORE_TLS, 
# Also define landcoverpy envs 