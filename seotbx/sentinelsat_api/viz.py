import logging
logger = logging.getLogger("seotbx.sentinelsat.viz")
import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt
import geoplot.crs as gcrs



def plot_footprints(products_gdf):
    world = gpd.read_file(
        gpd.datasets.get_path('naturalearth_lowres')
    )
    print(gpd.datasets.available)
    ax = gplt.polyplot(world, projection=gcrs.WebMercator())
    gplt.polyplot(products_gdf, projection=gcrs.WebMercator(), edgecolor='r', ax=ax)

    plt.show()
    toto = 0
