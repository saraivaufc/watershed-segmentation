import argparse
import uuid

from osgeo import osr, ogr, gdal

ogr.UseExceptions()


class Polygonize():
    @staticmethod
    def process(input_source):
        raster_proj = osr.SpatialReference()
        raster_proj.ImportFromWkt(input_source.GetProjection())

        driver = ogr.GetDriverByName('MEMORY')
        data_source = driver.CreateDataSource(uuid.uuid4().hex)
        layer = data_source.CreateLayer('polygonized', srs=raster_proj)

        id_field = ogr.FieldDefn('ID', ogr.OFTInteger)
        layer.CreateField(id_field)

        band = input_source.GetRasterBand(1)

        gdal.Polygonize(band, None, layer, 0, [])
        layer.ResetReading()

        return data_source


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?',
                        help="Input image e.g mosaic.tif")
    parser.add_argument('--output', type=str, nargs='?',
                        help="Output segmentation e.g output.shp")
    args = parser.parse_args()

    input_filename = args.input
    output_filename = args.output

    raster_dataset = gdal.Open(input_filename)

    polygonized_dataset = Polygonize.process(raster_dataset)

    driver = ogr.GetDriverByName('GPKG')
    output_dataset = driver.CreateDataSource(output_filename)
    output_layer = output_dataset.CopyLayer(polygonized_dataset.GetLayer(),
                                            "output")
    output_dataset = None
