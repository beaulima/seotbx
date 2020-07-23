import logging
import seotbx
import datetime
import os
import glob

logger = logging.getLogger("seotbx.sentinelsat.apps.download_sentinel1")

MONTREAL_CENTRE_LAT = 45.508888
MONTREAL_CENTRE_LONG = -73.561668

WKT_MTL_CENTRE = f'POINT({MONTREAL_CENTRE_LONG} {MONTREAL_CENTRE_LAT})'
WKT_OTT_MTL = "POLYGON((-75.83782093982082 45.31467961684288,-73.1250248504279 45.31467961684288,-73.1250248504279 45.77714479622745,-75.83782093982082 45.77714479622745,-75.83782093982082 45.31467961684288))"

def parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="download sentinel1 from scihub")
    ap.add_argument("-u", "--scihub_username", default=None, type=str, help="scihub user name")
    ap.add_argument("-p", "--scihub_passwd", default=None, type=str, help="scihub password name")
    ap.add_argument("-s", "--save-dir", default=None, type=str, help="download directory", required=True)
    ap.add_argument("-t0", "--date_start", default=None, type=str, help="query date start", required=True)
    ap.add_argument("-tf", "--date_end", default=None, type=str, help="query date end", required=True)
    ap.add_argument("-wkt", "--wkt_roi", default=WKT_OTT_MTL, type=str, help="region of interest")
    ap.add_argument("-pt", "--product_type", default="SLC", choices=["SLC", "GRD", "OCN"], help="product type")
    ap.add_argument("-od", "--orbit_direction", default="ASCENDING", choices=["ASCENDING", "DESCENDING"],
                    help="orbit direction")
    ap.add_argument("-op", "--operational_mode", default="IW", choices=["IW", "SM", "EW", "WV"],
                    help="sensor operational mode")

def application_func(args):
    b_download = True
    sentinel_api_obj = seotbx.sentinelsat_api.utils.get_sentinel_api(user=args.scihub_username,
                                                                password=args.scihub_passwd)
    wkt_roi = args.wkt_roi
    product_type = args.product_type
    orbit_direction = args.orbit_direction
    operational_mode = args.operational_mode
    date_start = datetime.datetime.strptime(args.date_start, "%d%b%Y")
    date_end = datetime.datetime.strptime(args.date_end, "%d%b%Y")
    candidates = seotbx.sentinelsat_api.utils.sentinel1_candidate(sentinel_api_obj=sentinel_api_obj,
                                                    roi=wkt_roi,
                                                    date=(date_start, date_end),
                                                    product_type=product_type,
                                                    orbit_direction=orbit_direction,
                                                    operational_mode=operational_mode
                                               )
    seotbx.sentinelsat_api.utils.log_candidates_info(sentinel_api_obj, candidates)
    
    input_dir = args.save_dir
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    os.chdir(input_dir)

    product_names = [os.path.basename(value) for value in glob.glob(input_dir + "/*.zip")]
    download_candidates = {}
    for key, item in candidates.items():
        title = item['title']
        if not f'{title}.zip' in product_names:
            download_candidates[key] = item
    seotbx.sentinelsat_api.utils.log_candidates_info(sentinel_api_obj, download_candidates)

    if b_download:
        sentinel_api_obj.download_all(download_candidates)

    logger.info("...done")
    return


def registering_application():
    from seotbx.apps import registering_application as ra
    ra("download_sentinel1", parser_func, application_func)