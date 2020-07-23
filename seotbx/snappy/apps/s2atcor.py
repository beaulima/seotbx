import logging
import os
import seotbx
import zipfile
logger = logging.getLogger("seotbx.snappy.apps.s2atcor")

def s2atcor(args, b_saved_only_10m=True, b_delete_srtm=False):

    seotbx.logger.info("Atmospheric correction on Sentinel2 with bands extraction in tif")
    if len(args.args) is 0:
        raise Exception("The arguments list is empty!")
    file_path = args.args[0]
    if not os.path.exists(file_path):
        raise Exception(f"File not found: {file_path}")

    try:
        workspace = os.environ['WORKSPACE']
    except:
        workspace = os.path.dirname(file_path)

    if workspace == "":
        workspace = "./"

    os.chdir(workspace)
    output_directory = workspace
    try:
        # the file must be unzip first
        filezip = zipfile.ZipFile(file_path)
        logging.info("Unzipping")
        filezip.extractall(workspace)
        filezip.close()
        if ".SAFE.zip" in file_path:
            file_path = file_path.replace(".SAFE.zip", '.SAFE')
        elif ".zip" in file_path:
            file_path = file_path.replace('.zip', '.SAFE')
    except:
        logging.info("Processing")

    key_name = os.path.basename(file_path)
    key_name = key_name[0:26]
    basedir = os.path.dirname(file_path)

    key_name = key_name.replace("MSIL1C", "MSIL2A")

    import glob
    files = glob.glob(f'{os.path.join(basedir, key_name)}*')
    # removing all files L2
    import shutil
    for file in files:
        seotbx.logger.info(f"Removing: {file}")
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)
        seotbx.logger.info(f"Deleting: {file}")

    command_args = []
    targetProductFile = f"tmp_{key_name}"
    command_args.append("Sen2Cor280")
    command_args.append(f"-SsourceProduct={file_path}")
    command_args.append("-Presolution=ALL")
    command_args.append(f"-PdemDirectory={output_directory}")
    command_args.append("-PgenerateDEMoutput=TRUE")
    command_args.append("-PgenerateTCIoutput=TRUE")
    command_args.append("-PgenerateDDVoutput=TRUE")
    command_args.append("-PnbThreads=8")
    command_args.append("-PDEMTerrainCorrection=TRUE")
    command_args.append("-PcirrusCorrection=TRUE")
    command_args.append("-PcirrusCorrection=TRUE")
    #command_args.append(f"-PtargetProductFile={targetProductFile}")
    seotbx.snappy.core.run_gpt_base(command_args)

    files = glob.glob(f'{os.path.join(basedir, key_name)}*')
    filepath_input = files[0]
    filepath_output = filepath_input
    if b_saved_only_10m:
         command_args = []
         command_args.append("BandSelect")
         command_args.append("-Ssource=%s" % filepath_input)
         command_args.append("-PsourceBands=%s,%s,%s,%s" % ("B2", "B3", "B4", "B8"))
         filepath_output = filepath_input.replace(".SAFE", "_10m.tif")
         command_args.append("-t")
         command_args.append(filepath_output)
         command_args.append("-f")
         command_args.append("GeoTIFF-BigTIFF")

         seotbx.snappy.core.run_gpt_base(command_args)
         logging.debug("Saved: %s" % (filepath_output))

    # Remove subimages
    files = glob.glob(f'{os.path.join(basedir, "target")}*')
    # removing all files L2
    import shutil
    for file in files:
        seotbx.logger.debug(f"Removing: {file}")
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)
        seotbx.logger.info(f"Deleting: {file}")

    if b_delete_srtm:
        files = glob.glob(f'{os.path.join(basedir, "srtm")}*')
        # removing all files L2
        import shutil
        for file in files:
            seotbx.logger.debug(f"Removing: {file}")
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)
            seotbx.logger.info(f"Deleting: {file}")

    return filepath_output


def parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="atmospheric correction on Sentinel2 with bands extraction in tif")
    ap.add_argument('args', nargs=argparse.REMAINDER)


def application_func(args):
    seotbx.snappy.utils.check_esa_snap_installation()
    s2atcor(args)

def registering_application():
    from seotbx.apps import registering_application as ra
    ra("s2atcor", parser_func, application_func)