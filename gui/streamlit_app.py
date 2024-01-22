import json
import folium
import base64
from folium.plugins import LocateControl, MousePosition, Draw, FloatImage
from streamlit_folium import folium_static
import streamlit as st
from PIL import Image
import rasterio
from folium.raster_layers import ImageOverlay
from json.decoder import JSONDecodeError
from seldon_core.seldon_client import SeldonClient
import requests
from pyproj import CRS, Proj, Transformer

st.set_page_config(layout="wide")

MAP_WIDTH_PX = 1500
MAP_HEIGHT_PX = 800
gateway_endpoint = "localhost:8080" # Modify it with the cluster's Istio Gateway endpoint

wgs84_crs = CRS.from_epsg(4326)

def download_raster(url: str) -> None:
    response = requests.get(url, stream=True)
    with open("raster.tif", "wb") as handle:
        for data in response.iter_content():
            handle.write(data)

def generate_map() -> folium.Map:
    map_folium = folium.Map(
                location=[0, 0],
                zoom_start=2
            )
    
    Draw(
        export=True, 
        filename="drawn_geometry.geojson",
        draw_options={
            "circle":False,
            "circlemarker": False,
            "marker": False,
            "rectangle": {
                "shapeOptions": {
                    "color": "yellow",
                    "fill": True,
                    "fillColor": "yellow",
                    "opacity": "1",
                    "fillOpacity": "0.3"
                }
            },
            "polyline": False,
            "polygon": {
                "allowIntersection": False,
                "showArea": True,
                "showLength": True,
                "metric": ["km", "m"],
                "feet": False,
                "shapeOptions": {
                    "color": "yellow",
                    "fill": True,
                    "fillColor": "yellow",
                    "opacity": "1",
                    "fillOpacity": "0.3"
                }
            }
        }
    ).add_to(map_folium)
    
    LocateControl().add_to(map_folium)
    MousePosition().add_to(map_folium)

    map_layers_dict = {
        "World Street Map": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", 
        "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Google Maps": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        "Google Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "Google Terrain": 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        "Google Satellite Hybrid": 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'
    }

    for layer in map_layers_dict:
        folium.TileLayer(
            tiles=map_layers_dict[layer],
            attr=layer,
            name=layer
        ).add_to(map_folium)
    return map_folium


map_folium = generate_map()

file_geojson = st.sidebar.file_uploader(
    "Upload geoJSON for prediction AOI", type=["geojson"], accept_multiple_files=False
    )

geojson = None
if file_geojson is not None:
    try:
        geojson = json.load(file_geojson)

        if (len(geojson["features"]) != 1) and (geojson["features"][0]["geometry"]["type"] != "Polygon"):
            raise JSONDecodeError
        
        folium.GeoJson(
            geojson,
            name="AOI",
            style_function=lambda x: {
                "color": "yellow",
                "fill": True,
                "fillColor": "yellow", 
                "opacity": "1",
                "fillOpacity": "0"
            },
        ).add_to(map_folium)
        
    except JSONDecodeError:
        st.error("""WRONG FORMAT FOR POLYGON. The uploaded GeoJSON file must follow the [GeoJSON format](https://geojson.org/).
                    For an example of correct format, please export a polygon using the functionality available in the map presentation modules.
                    """)
        st.stop()

multiple_years = st.sidebar.checkbox("Predict over multiple years", value=False)
if multiple_years:
    years = st.sidebar.multiselect('Select years', ["2020", "2021", "2022", "2023"])
else:
    years = [st.sidebar.selectbox('Select year', ["2020", "2021", "2022", "2023"], index=3)]

colors = [(32, 42, 174, 0), (250, 0, 0, 255), (255, 255, 76, 255), (255, 187, 34, 255), (0, 50, 200, 255), (0, 150, 160, 255), (240, 150, 255, 255), (0, 120, 0, 255), (100, 140, 0, 255), (180, 180, 180, 255)]

if st.sidebar.button("Predict", type="primary") and geojson is not None:
    with st.spinner("Predicting..."):
        for year in years:
            sc = SeldonClient(deployment_name=f"landcover-seldon-{year}", namespace="mlops-seldon")
            res = sc.predict(transport="grpc",gateway="istio",gateway_endpoint=gateway_endpoint,raw_data={"strData":json.dumps(geojson)})
            download_url = res.response["jsonData"]["result"]
            download_raster(download_url)

            with rasterio.open("./raster.tif") as raster_file:
                band = raster_file.read(1)
                bounds = raster_file.bounds
                local_crs = raster_file.crs
                transformer = Transformer.from_crs(local_crs, wgs84_crs, always_xy=True)
                left_lon, bottom_lat = transformer.transform(bounds.left, bounds.bottom)
                right_lon, top_lat = transformer.transform(bounds.right, bounds.top)

            ImageOverlay(
                image=band,
                name=f"land cover prediction in {year}",
                bounds=[[bottom_lat, left_lon], [top_lat, right_lon]],
                colormap=lambda x: colors[x],
                opacity=1
            ).add_to(map_folium)

    im_width, _ = Image.open("legend.png").size
    im_width_percent = im_width * 100 / MAP_WIDTH_PX
    with open("legend.png", 'rb') as lf:
        b64_content = base64.b64encode(lf.read()).decode('utf-8')
    FloatImage('data:image/png;base64,{}'.format(b64_content), bottom=5.15, left=99.85-im_width_percent).add_to(map_folium)

folium.LayerControl(position="bottomleft").add_to(map_folium)
map_folium.fit_bounds(map_folium.get_bounds())
folium_static(map_folium, height=MAP_HEIGHT_PX, width=MAP_WIDTH_PX)


