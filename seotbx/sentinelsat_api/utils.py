import logging
logger = logging.getLogger("seotbx.sentinelsat.utils")
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date

SCIHUB_API_URL = "https://scihub.copernicus.eu/dhus"

def get_sentinel_api(user=None, password=None, api_url=SCIHUB_API_URL):
    api = SentinelAPI(user=user, password=password, api_url=SCIHUB_API_URL)
    return api


def sentinel1_candidate(sentinel_api_obj, roi, date, product_type, orbit_direction, operational_mode):
    candidate = sentinel_api_obj.query(area=roi,
                                               date=date,
                                               producttype=product_type,
                                               orbitdirection=orbit_direction,
                                               sensoroperationalmode=operational_mode)

    return candidate

def log_candidates_info(sentinel_api_obj, candidates):
    title_found_sum = 0
    for key, value in candidates.items():
        for k, v in value.items():
            if k == 'title':
                title_info = v
                title_found_sum += 1
            elif k == 'size':
                logger.info("title: " + title_info + " | " + v)
    logger.info(f"Total found {title_found_sum} title of {sentinel_api_obj.get_products_size(candidates)} GB")