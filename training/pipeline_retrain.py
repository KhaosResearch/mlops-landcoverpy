
import json
import os
import random
import yaml
from os.path import join
from pathlib import Path
from shutil import rmtree

import joblib
import numpy as np
import pandas as pd
import mlflow
from kubernetes import client, config
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from sklearn.ensemble import RandomForestClassifier
from prefect import flow, task, unmapped
from prefect.blocks.kubernetes import KubernetesClusterConfig

from landcoverpy.aster import get_dem_from_tile
from landcoverpy.composite import _create_composite, _get_composite
from landcoverpy.config import settings
from landcoverpy.exceptions import NoAsterException
from landcoverpy.execution_mode import ExecutionMode
from landcoverpy.minio import MinioConnection
from landcoverpy.mongo import MongoConnection
from landcoverpy.utilities.confusion_matrix import compute_confusion_matrix
from landcoverpy.utilities.geometries import _group_polygons_by_tile
from landcoverpy.utilities.raster import (
    _download_sample_band_by_tile,
    _filter_rasters_paths_by_features_used,
    _get_kwargs_raster,
    _get_product_rasters_paths,
    _get_raster_filename_from_path,
    _get_raster_name_from_path,
    _read_raster,
)
from landcoverpy.utilities.utils import (
    _mask_polygons_by_tile,
    get_products_by_tile_and_date,
    get_season_dict,
)

from landcover_mlflow_wrapper import LancoverMlflowWrapper

@flow
def retraining_flow(data_bucket: str, data_folder: str, model_name: str, test_size: float):
    geojsons_dict = ingest_data(data_bucket, data_folder)
    polygons_per_tile = group_data_by_tile(geojsons_dict)
    used_columns = get_used_columns_by_model(model_name)
    tile_datasets = process_tile.map(list(polygons_per_tile.keys()), list(polygons_per_tile.values()), unmapped(used_columns))
    merged_dataset = merge_tile_datasets(tile_datasets)
    retrain_model(merged_dataset, model_name, test_size, used_columns)
    deploy_best_model(model_name, wait_for=[retrain_model])

@task
def deploy_best_model(model_name):

    mlflow_client = mlflow.client.MlflowClient()

    model_versions_metadata = mlflow_client.search_model_versions(f"name='landcoverpy'").to_list()

    production_run_id = next(filter(lambda x: x.current_stage == "Production", model_versions_metadata)).run_id
    production_accuracy = mlflow_client.get_metric_history(production_run_id, key="accuracy")[0].value


    latest_run_id = max(model_versions_metadata, key=lambda x: int(x.version)).run_id
    latest_version = max(model_versions_metadata, key=lambda x: int(x.version)).version
    latest_accuracy = mlflow_client.get_metric_history(latest_run_id, key="accuracy")[0].value

    is_new_model_better =  latest_accuracy > production_accuracy

    if is_new_model_better:

        print("deploying retrained model")

        cluster_config_block = KubernetesClusterConfig.load("k8s-config")
        config.load_kube_config_from_dict(cluster_config_block.config)
        custom_api = client.CustomObjectsApi()
        with open("files/deploy_landcover.yaml", 'r') as stream:
            deployment_yaml = yaml.safe_load(stream)

        try:

            mlflow_client.transition_model_version_stage(model_name, latest_version, stage="Production")

            custom_api.create_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1alpha2",
                namespace="mlops-seldon",
                plural="seldondeployments",
                body=deployment_yaml
            )
        except:
            existing_deployment_yaml = custom_api.get_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1alpha2",
                namespace="mlops-seldon",
                plural="seldondeployments",
                name=deployment_yaml["metadata"]["name"],
            )

            custom_api.delete_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1alpha2",
                namespace="mlops-seldon",
                plural="seldondeployments",
                name=existing_deployment_yaml["metadata"]["name"],
            )

            custom_api.create_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1alpha2",
                namespace="mlops-seldon",
                plural="seldondeployments",
                body=deployment_yaml
            )

@task
def retrain_model(new_data: pd.DataFrame, model_name: str, test_size: float, used_columns: str):
    model_uri = f"models:/{model_name}/production"
    download_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=settings.TMP_DIR)
    production_testing_dataset = pd.read_csv(os.path.join(download_path, "artifacts", "testing_dataset.csv"))
    production_training_dataset = pd.read_csv(os.path.join(download_path, "artifacts", "training_dataset.csv"))

    new_data = new_data.replace([np.inf, -np.inf], np.nan)
    new_data = new_data.fillna(np.nan)
    new_data = new_data.dropna()
    new_data["location"] = list(zip(new_data["latitude"], new_data["longitude"]))
    new_data = new_data.drop(
        [
            "latitude",
            "longitude",
            "spring_product_name",
            "autumn_product_name",
            "summer_product_name",
        ],
        axis=1,
    )

    locations = list(set(new_data["location"]))
    random.shuffle(locations)

    n_locations = len(locations)

    test_locations = locations[:int(test_size*n_locations) + 1]
    train_locations = locations[int(test_size*n_locations) + 1:]

    train_df = new_data[new_data["location"].isin(train_locations)]
    test_df = new_data[new_data["location"].isin(test_locations)]
    test_df = test_df.drop_duplicates(subset=["location"]).reset_index(
        drop=True
    )

    test_df = test_df.drop(["location"], axis=1)
    train_df = train_df.drop(["location"], axis=1)

    test_df = pd.concat([test_df, production_testing_dataset], axis=0)
    train_df = pd.concat([train_df, production_training_dataset], axis=0)

    X_test = test_df.drop(["class"], axis=1)
    X_train = train_df.drop(["class"], axis=1)
    y_test = test_df["class"]
    y_train = train_df["class"]

    # Train model
    n_jobs = 1
    clf = RandomForestClassifier(n_jobs=n_jobs)
    X_train = X_train[used_columns]
    print(X_train)
    clf.fit(X_train, y_train)
    X_test = X_test[used_columns]
    y_true = clf.predict(X_test)

    labels = production_training_dataset["class"].unique()

    confusion_image_filename = "confusion_matrix.png"
    out_image_path = join(settings.TMP_DIR, confusion_image_filename)
    compute_confusion_matrix(y_true, y_test, labels, out_image_path=out_image_path)

    model_name = "model.joblib"
    model_path = join(settings.TMP_DIR, model_name)
    joblib.dump(clf, model_path)

    # Save training and test dataset to minio
    training_data_name = "training_dataset.csv"
    training_data_path = join(settings.TMP_DIR, training_data_name)
    pd.concat([X_train, y_train], axis=1).to_csv(training_data_path, index=False)

    testing_data_name = "testing_dataset.csv"
    testing_data_path = join(settings.TMP_DIR, testing_data_name)
    pd.concat([X_test, y_test], axis=1).to_csv(testing_data_path, index=False)

    model_metadata = {
        "model": str(type(clf)),
        "n_jobs": n_jobs,
        "used_columns": list(used_columns),
        "classes": list(labels),
        "test_size": str(test_size)
    }

    model_metadata_name = "metadata.json"
    model_metadata_path = join(settings.TMP_DIR, model_metadata_name)

    with open(model_metadata_path, "w") as f:
        json.dump(model_metadata, f)

    experiment_id = mlflow.get_experiment_by_name("landcoverpy").experiment_id

    with mlflow.start_run(experiment_id=experiment_id):

        input_schema = Schema([ColSpec("string","input geometry")])
        output_schema = Schema([ColSpec("integer")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_example = {'input geometry': '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[[9.80232,44.10784],[9.79774,44.09437],[9.84088,44.08853],[9.87776,44.10170],[9.84400,44.12190],[9.80232,44.10784]]],"type":"Polygon"}}]}'}

        mlflow.pyfunc.log_model(artifact_path="model",
                                registered_model_name="landcoverpy",
                                code_path=["landcover_mlflow_wrapper.py"],
                                pip_requirements=["landcoverpy","mlflow==2.3.1"],
                                python_model=LancoverMlflowWrapper(),
                                artifacts={
                                    "model_file": model_path, 
                                    "confusion_matrix": out_image_path, 
                                    "training_data": training_data_path, 
                                    "testing_data": testing_data_path, 
                                    "metadata_file": model_metadata_path,
                                },
                                signature=signature,
                                input_example=input_example
        )
        
        mlflow.log_params(model_metadata)

        metrics = confusion_matrix_to_metrics(out_image_path.replace(".png", ".csv"))
        mlflow.log_metrics(metrics)    

@task
def get_used_columns_by_model(model_name: str) -> list[str]:
    model_uri = f"models:/{model_name}/production"
    download_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=settings.TMP_DIR)

    metadata_file = os.path.join(download_path, "artifacts", "metadata.json")

    with open(metadata_file) as f:
        metadata = json.load(f)
    used_columns = metadata["used_columns"]
    return used_columns

@task
def ingest_data(bucket: str, folder: str):
    geojson_dicts =  {}
    minio_client = MinioConnection()
    
    geojson_files_cursor = minio_client.list_objects(
        bucket_name=bucket,
        prefix=join(folder,"")
    )

    # For each image
    for geojson_files in geojson_files_cursor:
        geojson_files_path = geojson_files.object_name

        response = minio_client.get_object(
            bucket_name=bucket,
            object_name=geojson_files_path
        )

        classification_label = geojson_files_path.split("/")[-1].split(".")[-2]
        geojson_dicts[classification_label] = json.loads(response.data)

    return geojson_dicts

@task
def group_data_by_tile(geojson_dicts: dict[str,dict]) -> dict[str,dict]:
    polygons_per_tile = _group_polygons_by_tile(geojson_dicts)
    return polygons_per_tile

@task
def merge_tile_datasets(tile_datasets: list[pd.DataFrame]) -> pd.DataFrame:
    final_df = pd.concat(tile_datasets, axis=0)
    return final_df

@task
def process_tile(tile, polygons_in_tile, used_columns=None):

    if not Path(settings.TMP_DIR).exists():
        Path.mkdir(Path(settings.TMP_DIR))

    seasons = get_season_dict()

    minio_client = MinioConnection()
    mongo_client = MongoConnection()
    mongo_products_collection = mongo_client.get_collection_object()

    # Names of the indexes that are taken into account
    indexes_used = [
        "cri1",
        "ri",
        "evi2",
        "mndwi",
        "moisture",
        "ndyi",
        "ndre",
        "ndvi",
        "osavi",
    ]
    # Name of the sentinel bands that are ignored
    skip_bands = ["tci", "scl"]
    # Ranges for normalization of each raster
    normalize_range = {"slope": (0, 70), "aspect": (0, 360), "dem": (0, 2000)}

    print(f"Working in tile {tile}")
    # Mongo query for obtaining valid products
    max_cloud_percentage = settings.MAX_CLOUD

    spring_start, spring_end = seasons["spring"]
    product_metadata_cursor_spring = get_products_by_tile_and_date(
        tile, mongo_products_collection, spring_start, spring_end, max_cloud_percentage
    )

    summer_start, summer_end = seasons["summer"]
    product_metadata_cursor_summer = get_products_by_tile_and_date(
        tile, mongo_products_collection, summer_start, summer_end, max_cloud_percentage
    )

    autumn_start, autumn_end = seasons["autumn"]
    product_metadata_cursor_autumn = get_products_by_tile_and_date(
        tile, mongo_products_collection, autumn_start, autumn_end, max_cloud_percentage
    )

    product_per_season = {
        "spring": list(product_metadata_cursor_spring)[-settings.MAX_PRODUCTS_COMPOSITE:],
        "autumn": list(product_metadata_cursor_autumn)[:settings.MAX_PRODUCTS_COMPOSITE],
        "summer": list(product_metadata_cursor_summer)[-settings.MAX_PRODUCTS_COMPOSITE:],
    }

    if (
        len(product_per_season["spring"]) == 0
        or len(product_per_season["autumn"]) == 0
        or len(product_per_season["summer"]) == 0
    ):
        return pd.DataFrame(columns=used_columns)

    # Dataframe for storing data of a tile
    tile_df = None

    dems_raster_names = [
        "slope",
        "aspect",
        "dem",
    ]
    
    for dem_name in dems_raster_names:
        # Add dem and aspect data
        if dem_name in used_columns:
            
            try:
                dem_path = get_dem_from_tile(
                    tile, mongo_products_collection, minio_client, dem_name
                )
            except NoAsterException:
                return pd.DataFrame(columns=used_columns) 

            # Predict doesnt work yet in some specific tiles where tile is not fully contained in aster rasters
            kwargs = _get_kwargs_raster(dem_path)
            crop_mask, label_lon_lat = _mask_polygons_by_tile(polygons_in_tile, kwargs)

            band_normalize_range = normalize_range.get(dem_name, None)
            raster, _ = _read_raster(
                band_path=dem_path,
                rescale=True,
                normalize_range=band_normalize_range,
            )
            raster_masked = np.ma.masked_array(raster, mask=crop_mask)
            raster_masked = np.ma.compressed(raster_masked).flatten()
            raster_df = pd.DataFrame({dem_name: raster_masked})
            tile_df = pd.concat([tile_df, raster_df], axis=1)

    # Get crop mask for sentinel rasters and dataset labeled with database points in tile
    band_path = _download_sample_band_by_tile(tile, minio_client, mongo_products_collection)
    kwargs = _get_kwargs_raster(band_path)

    crop_mask, label_lon_lat = _mask_polygons_by_tile(polygons_in_tile, kwargs)

    for season, products_metadata in product_per_season.items():
        print(season)
        bucket_products = settings.MINIO_BUCKET_NAME_PRODUCTS
        bucket_composites = settings.MINIO_BUCKET_NAME_COMPOSITES
        current_bucket = None

        if len(products_metadata) == 0:
            return pd.DataFrame(columns=used_columns)

        elif len(products_metadata) == 1:
            product_metadata = products_metadata[0]
            current_bucket = bucket_products
        else:
            # If there are multiple products for one season, use a composite.
            mongo_client.set_collection(settings.MONGO_COMPOSITES_COLLECTION)
            mongo_composites_collection = mongo_client.get_collection_object()
            products_metadata_list = list(products_metadata)
            product_metadata = _get_composite(
                products_metadata_list, mongo_composites_collection, ExecutionMode.TRAINING
            )
            if product_metadata is None:
                _create_composite(
                    products_metadata_list,
                    minio_client,
                    bucket_products,
                    bucket_composites,
                    mongo_composites_collection,
                    ExecutionMode.TRAINING
                )
                product_metadata = _get_composite(
                    products_metadata_list, mongo_composites_collection, ExecutionMode.TRAINING
                )
            current_bucket = bucket_composites

        product_name = product_metadata["title"]

        # For validate dataset geometries, the product name is added.
        raster_product_name = np.full_like(
            raster_masked, product_name, dtype=object
        )
        raster_df = pd.DataFrame({f"{season}_product_name": raster_product_name})
        tile_df = pd.concat([tile_df, raster_df], axis=1)

        (rasters_paths, is_band) = _get_product_rasters_paths(
            product_metadata, minio_client, current_bucket
        )

        (rasters_paths, is_band) = _filter_rasters_paths_by_features_used(
                rasters_paths, is_band, used_columns, season
            )

        temp_product_folder = Path(settings.TMP_DIR, product_name + ".SAFE")
        if not temp_product_folder.exists():
            Path.mkdir(temp_product_folder)
        print(f"Processing product {product_name}")

        # Read bands and indexes.
        already_read = []
        for i, raster_path in enumerate(rasters_paths):
            raster_filename = _get_raster_filename_from_path(raster_path)
            raster_name = _get_raster_name_from_path(raster_path)
            temp_path = Path(temp_product_folder, raster_filename)

            # Only keep bands and indexes in indexes_used
            if (not is_band[i]) and (
                not any(
                    raster_name.upper() == index_used.upper()
                    for index_used in indexes_used
                )
            ):
                continue
            # Skip bands in skip_bands
            if is_band[i] and any(
                raster_name.upper() == band_skipped.upper()
                for band_skipped in skip_bands
            ):
                continue
            # Read only the first band to avoid duplication of different spatial resolution
            if any(
                raster_name.upper() == read_raster.upper()
                for read_raster in already_read
            ):
                continue
            already_read.append(raster_name)

            print(f"Downloading raster {raster_name} from minio into {temp_path}")
            minio_client.fget_object(
                bucket_name=current_bucket,
                object_name=raster_path,
                file_path=str(temp_path),
            )
            kwargs = _get_kwargs_raster(str(temp_path))
            spatial_resolution = kwargs["transform"][0]
            if spatial_resolution == 10:
                kwargs_10m = kwargs

            band_normalize_range = normalize_range.get(raster_name, None)
            if is_band[i] and (band_normalize_range is None):
                band_normalize_range = (0, 7000)

            raster = _read_raster(
                band_path=temp_path,
                rescale=True,
                normalize_range=band_normalize_range,
            )
            raster_masked = np.ma.masked_array(raster[0], mask=crop_mask)
            raster_masked = np.ma.compressed(raster_masked)

            raster_df = pd.DataFrame({f"{season}_{raster_name}": raster_masked})

            tile_df = pd.concat([tile_df, raster_df], axis=1)



    for index, label in enumerate(["class", "longitude", "latitude", "forest_type"]):

        raster_masked = np.ma.masked_array(label_lon_lat[:, :, index], mask=crop_mask)
        raster_masked = np.ma.compressed(raster_masked).flatten()
        raster_df = pd.DataFrame({label: raster_masked})
        tile_df = pd.concat([tile_df, raster_df], axis=1)

    print("Dataframe information:")
    print(tile_df.info())

    for path in Path(settings.TMP_DIR).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)
    
    print(tile_df)
    
    return tile_df

def confusion_matrix_to_metrics(confusion_matrix_path: str) -> dict:

    confusion_matrix = pd.read_csv(confusion_matrix_path, index_col=0)

    classes = confusion_matrix.columns.tolist()
    matrix_array = confusion_matrix.values.astype(int)
    metrics = {}

    for i, class_name in enumerate(classes):
        values = matrix_array[i]

        tp = values[i]
        fn = np.sum(values) - tp
        fp = np.sum(matrix_array[:, i]) - tp

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

        metrics[f'precision_{class_name}'] = precision
        metrics[f'recall_{class_name}'] = recall
        metrics[f'f1_score_{class_name}'] = f1_score

    total_tp = np.sum(np.diagonal(matrix_array))
    total_fp = np.sum(matrix_array, axis=0) - np.diagonal(matrix_array)

    global_accuracy = total_tp / (total_tp + np.sum(total_fp))
    metrics['accuracy'] = global_accuracy

    return metrics