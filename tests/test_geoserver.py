import sys
import os
import traceback
from geoserver.catalog import Catalog
# the last "/" is very important

try:
   GEOSERVER_SERVICE_URL=os.environ["GEOSERVER_SERVICE_URL"]
except KeyError:
    print("Please set the environment variable GEOSERVER_SERVICE_URL")
    sys.exit(1)
try:
   GEOSERVER_USERNAME=os.environ["GEOSERVER_USERNAME"]
except KeyError:
    print("Please set the environment variable GEOSERVER_USERNAME")
    sys.exit(1)
try:
   GEOSERVER_PASSWORD=os.environ["GEOSERVER_PASSWORD"]
except KeyError:
    print("Please set the environment variable GEOSERVER_PASSWORD")
    sys.exit(1)

geoserver_catalog = Catalog(service_url=GEOSERVER_SERVICE_URL,
                            username=GEOSERVER_USERNAME,
                            password=GEOSERVER_PASSWORD)
try:
    results = geoserver_catalog.create_workspace(name="TOTO")
except AssertionError:
    _, _, tb = sys.exc_info()
    traceback.print_tb(tb)  # Fixed format
    tb_info = traceback.extract_tb(tb)
    filename, line, func, text = tb_info[-1]
    print(f"An error occurred on line {line} in statement {text}")
    sys.exit(1)
