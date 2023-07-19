import random

from seldon_core.seldon_client import SeldonClient
from shapely.geometry import Polygon, MultiPolygon, shape


def generate_geometry_within_aoi(aoi_multipolygon: MultiPolygon, max_area: float):

    generated_polygon = MultiPolygon()

    while not ( generated_polygon.is_valid
               and generated_polygon.is_simple
               and generated_polygon.within(aoi_multipolygon)
               and generated_polygon.envelope.area <= max_area
               and (generated_polygon.bounds[2] - generated_polygon.bounds[0]) / (generated_polygon.bounds[3] - generated_polygon.bounds[1]) > 0.5
               and (generated_polygon.bounds[2] - generated_polygon.bounds[0]) / (generated_polygon.bounds[3] - generated_polygon.bounds[1]) < 2
    ):
        aoi_polygon = aoi_multipolygon.geoms[random.randint(0, len(aoi_multipolygon.geoms)-1)]

        x_coords, y_coords = zip(*aoi_polygon.boundary.coords)
        min_x, min_y, max_x, max_y = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
        random_x = random.uniform(min_x, max_x)
        random_y = random.uniform(min_y, max_y)
        num_points = random.randint(4, 10)  # Random number of points for the polygon

        generated_polygon = Polygon(
            [(random.uniform(random_x, random_x + max_x*0.01), random.uniform(random_y, random_y + max_y*0.01)) for _ in range(num_points)]
        )

    return str([list(coord) for coord in generated_polygon.exterior.coords])


sc = SeldonClient(deployment_name="landcover-seldon", namespace="mlops-seldon")

aoi = {"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[[-6.965033606527584,37.636032133850904],[-1.2046161253377363,38.03129252466405],[0.0210046296753319,41.50772321456839],[0.9527892644194935,41.977286713071095],[-5.868199827129985,42.570315607441415],[-5.441888008908165,39.56272557126417],[-6.789033354488794,38.54309621492828],[-6.965033606527584,37.636032133850904]]],"type":"Polygon"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[[[1.8366176420281306,36.096662337684066],[4.04902365462857,36.215438772168596],[7.001124646305129,36.01670733594368],[9.382856001332101,36.22224818951517],[10.391344220439407,35.753409271904175],[9.92892656755339,37.14816370759756],[7.231490259044961,36.37639563517605],[3.917497080020752,36.68603459062649],[1.8366176420281306,36.096662337684066]]],"type":"Polygon"}}]}
aoi = MultiPolygon([shape(geometry["geometry"]) for geometry in aoi["features"]])

for i in range(1000):

    geometry = generate_geometry_within_aoi(aoi, 0.0005)
    print(geometry)

    raw_data = {"strData":'{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[' + geometry + '],"type":"Polygon"}}]}'} 
    res = sc.predict(transport="grpc",gateway="istio",gateway_endpoint="localhost:8080",raw_data=raw_data)
    print(res)
    print("\n")