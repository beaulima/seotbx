import logging
import seotbx
import datetime
import os
import glob
import snappy

logger = logging.getLogger("seotbx.snappy_api.sentinel1.s1preprocess")


def s1process_parser_func(subparsers, mode, argparse=None):
    
    ap = subparsers.add_parser(mode, help="preprocessing sentinel1")
    ap.add_argument("filename", type=str, help="sentinel1 file path (zip or .SAFE)")
    ap.add_argument("-cf", "--continue_on_fail", default=False, type=bool, help="Apply-Orbit-File: Continue operation on failure",
                    required=False)
    ap.add_argument("-ot", "--orbit_type", default="Sentinel Precise (Auto Download)",
                    choices=["Sentinel Precise (Auto Download)",
                             "Sentinel Restituted (Auto Download)",
                             "DORIS Preliminary POR (ENVISAT)",
                             "DORIS Precise VOR (ENVISAT) (Auto Download)",
                             "DELFT Precise (ENVISAT, ERS1&2) (Auto Download)",
                             "PRARE Precise (ERS1&2) (Auto Download)",
                             "Kompsat5 Precise"], help="product type",
                    required=False)
    ap.add_argument("-pd", "--poly_degree", default=3, type=int, help="Apply-Orbit-File: polynome degree for interpolation",
                    required=False)



def s1process_application_func(args):

    seotbx.snappy_api.utils.check_esa_snap_installation()
    filename = args.filename
    continue_on_fail = args.continue_on_fail
    orbit_type = args.orbit_type
    poly_degree = args.poly_degree
    try:
        product_input = snappy.ProductIO.readProduct(filename)
    except:
        raise Exception(f"cannot open product {filename}")

    output1 = seotbx.snappy_api.sentinel1.utils.apply_orbit_file(product_input=product_input,
                                                                 continue_on_fail=continue_on_fail,
                                                                 orbit_type=orbit_type,
                                                                 poly_degree=poly_degree)
    product_name = output1.getName() + ".dim"
    product_path = os.path.join("/misc/tmp", product_name)
    snappy.ProductIO.writeProduct(output1, product_path, 'BEAM-DIMAP')

    output2 = seotbx.snappy_api.sentinel1.utils.thermal_noise_removal(product_input=output1)

    product_name = output2.getName() + ".dim"
    product_path = os.path.join("/misc/tmp", product_name)
    snappy.ProductIO.writeProduct(output2, product_path, 'BEAM-DIMAP')

    output3 = seotbx.snappy_api.sentinel1.utils.topsar_deburst_SLC(product_input=output2)
    product_name = output3.getName()+".dim"
    product_path = os.path.join("/misc/tmp", product_name)
    snappy.ProductIO.writeProduct(output3, product_path, 'BEAM-DIMAP')

    output4 = seotbx.snappy_api.sentinel1.utils.calibration(product_input=output3)



    return
