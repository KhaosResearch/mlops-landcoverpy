FROM python:3.11.3

LABEL org.opencontainers.image.source=https://github.com/KhaosResearch/mlops-landcoverpy/
LABEL org.opencontainers.image.description="Landcover model deployment image"
LABEL org.opencontainers.image.licenses=MIT

WORKDIR /home
COPY ./LandcoverSeldonWrapper.py /home
COPY ./requirements.txt /home
RUN mkdir -p -v /home/files/tmp
RUN chown -R 8888 /home
RUN chmod -R 755 /home

RUN apt-get -y  update
RUN apt-get install -y  libgeos-dev

RUN pip install --only-binary :all: rasterio==1.3.7
RUN pip install --upgrade pip

# https://github.com/yaml/pyyaml/issues/724#issuecomment-1638636728
RUN pip install "cython<3.0.0" && pip install --no-build-isolation pyyaml==5.4.1

RUN pip install -r requirements.txt
# Port for GRPC
EXPOSE 5000
# Port for REST
EXPOSE 9000

# Fixed value for required mlflow and seldon environment variables (don't modify)
ENV MODEL_NAME="LandcoverSeldonWrapper"
ENV SERVICE_TYPE="MODEL"

# Default value for required mlflow and seldon environment variables
ENV MLFLOW_S3_IGNORE_TLS=""
ENV MLFLOW_S3_ENDPOINT_URL=""
ENV MLFLOW_TRACKING_URI=""
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""

# Landcoverpy environment variables
ENV MINIO_HOST=""
ENV MINIO_PORT=""
ENV MINIO_ACCESS_KEY=""
ENV MINIO_SECRET_KEY=""
ENV MINIO_BUCKET_NAME_PRODUCTS=""
ENV MINIO_BUCKET_NAME_ASTER=""
ENV MINIO_BUCKET_NAME_DEM=""
ENV MINIO_BUCKET_MODELS=""
ENV MINIO_BUCKET_CLASSIFICATIONS=""
ENV MINIO_DATA_FOLDER_NAME=""
ENV MONGO_HOST=""
ENV MONGO_PORT=""
ENV MONGO_USERNAME=""
ENV MONGO_PASSWORD=""
ENV MONGO_DB=""
ENV MONGO_PRODUCTS_COLLECTION=""
ENV TMP_DIR=""
ENV SPRING_START=""
ENV SPRING_END=""
ENV SUMMER_START=""
ENV SUMMER_END=""
ENV AUTUMN_START=""
ENV AUTUMN_END=""
ENV LAND_COVER_MODEL_FOLDER=""

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE
