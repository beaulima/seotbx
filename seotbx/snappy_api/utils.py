import logging
import numpy as np
import snappy
import os
logger = logging.getLogger("seotbx.snappy_api.apps.sentinel1")

def check_esa_snap_installation():
    SNAP_HOME = os.environ['SNAP_HOME']
    logger.info(f"SNAP installation: {SNAP_HOME}")
    logger.info(f"SNAPPY installation: {snappy.__file__}")
    snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()


def isSNAPprod(prod):
    return 'snap.core.datamodel.Product' in str(type(prod))

def getMinMax(current, minV, maxV):
    if current < minV:
        minV = current
    if current > maxV:
        maxV = current
    return [minV, maxV]


def getExtent(file1):
    ########
    ## Get corner coordinates of the ESA SNAP product (get extent)
    ########
    # int step - the step given in pixels
    step = 1
    minLon = 999.99

    myProd = readProd(file1)
    try:
        GeoPos = snappy.ProductUtils.createGeoBoundary(myProd, step)
    except RuntimeError as e:
        logger.info("\t".join(["getExtent",
                              "Error!!!, Probably file: '{0}' has *no* bands. Result of len(myProd.getBands()): '{1}'".format(
                                  file1, len(myProd.getBands()))]))
        logger.info("\t".join(["getExtent", "Error message: '{0}'".format(e)]))
        return [0.0, 0.0, 0.0, 0.0]
    maxLon = -minLon
    minLat = minLon
    maxLat = maxLon
    # TODO: probably there's better way to check min/max (?)
    for element in GeoPos:
        try:
            lon = element.getLon()
            [minLon, maxLon] = getMinMax(lon, minLon, maxLon)
        except NameError:
            pass
        try:
            # TODO: separate method to get min and max
            lat = element.getLat()
            [minLat, maxLat] = getMinMax(lat, minLat, maxLat)
        except NameError:
            pass
    myProd.dispose()
    return [minLon, maxLon, minLat, maxLat]


def getExtentStr(file1):
    array = getExtent(file1)
    for i in range(len(array)):
        array[i] = str(round(array[i], 2))
    return "\t".join(["Lon:", array[0], array[1], "Lat:", array[2], array[3]])


def readProd(file1):
    import snappy, os
    if isSNAPprod(file1):
        # input parameter is already a SNAP product
        return file1
    if os.path.isfile(file1):
        prod = snappy.ProductIO.readProduct(file1)
    elif os.path.exists(file1):
        logger.info("\t".join(["readProduct", str(file1), "is not a file!!!"]))
        prod = None
    else:
        logger.info("\t".join(["readProduct", str(file1), "does *NOT* exists"]))
        prod = None
    return prod

def getBand(name, source):

    bandsname = source.getBandNames()

    xSize0 = source.getSceneRasterWidth()
    ySize0 = source.getSceneRasterHeight()
    band = source.getBand(name)

    if band is None:
        return None
    band.getRasterHeight()
    xSize = band.getRasterWidth()
    ySize = band.getRasterHeight()
    buffer = np.zeros(xSize, dtype='int32')
    image = []
    for j in range(ySize):
        band.readPixels(0, j, xSize, 1, buffer)
        image.append(np.copy(buffer))
    image=np.vstack(image)
    return image



def getProductRes(file1):
    ##
    # Gets product resolution in geographical degrees
    ##
    precision = 7
    myProd = readProd(file1)
    height = float(myProd.getSceneRasterHeight())
    width = float(myProd.getSceneRasterWidth())
    myProd.dispose()
    #
    [minLon, maxLon, minLat, maxLat] = getExtent(file1)
    Lon = maxLon - minLon
    Lat = maxLat - minLat
    # TODO: THIS MUST BE FIXED!!!
    # Tested on 'test_TIFF.tif' file in *this* repository
    # For example: gdalinfo(test_TIFF.tif) shows me 'Pixel Size = (0.259366035461426,-0.316413879394531)'
    # but this method returns: '0.2074928, 0.1582069'
    return "{0}, {1}".format(round(Lon / width, precision), round(Lat / height, precision))


def getProductInfo(file1):
    import snappy
    from snappy import GPF
    from snappy import ProductIO

    prod = readProd(file1)
    bandNames = ''
    for i in prod.getBandNames():
        bandNames += "'{0}'".format(i)
    firstBand = prod.getBands()[0]
    width = firstBand.getRasterWidth()
    height = firstBand.getRasterHeight()
    prod.dispose()
    resolution = getProductRes(file1)
    return "Bands: {0}, width = {1}, height = {2}, resolution = {3}".format(bandNames, width, height, resolution)


logger = logging.getLogger("seotbx.snappy.utils")