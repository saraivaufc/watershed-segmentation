import argparse
import math
import multiprocessing
import threading
import uuid

import numpy as np
from osgeo import ogr, osr, gdal

THREADS = multiprocessing.cpu_count()

ogr.UseExceptions()

lock = threading.Lock()

stats_dict = {
    'min': lambda x: x.min(),
    'max': lambda x: x.max(),
    'mean': lambda x: x.mean(),
    'std': lambda x: x.std(),
    'sum': lambda x: x.sum(),
    'count': lambda x: x.count(),
    'area': lambda g: g.GetArea(),
    'perimeter': lambda g: g.Boundary().Length(),
    'compactness': lambda g: (4 * math.pi * g.GetArea()) / (
            g.Boundary().Length() ** 2)
}

matricial_stats = ['min', 'max', 'mean', 'std', 'sum', 'count']
vectorial_stats = ['area', 'perimeter', 'compactness']


class ZonalStatistics():
    @staticmethod
    def thread_function(shape_layer, raster_dataset, stats):
        lock.acquire()
        shape_feature = shape_layer.GetNextFeature()
        lock.release()
        while not shape_feature is None:
            zonal_stats_result = ZonalStatistics.zonal_stats(shape_feature,
                                                             raster_dataset,
                                                             stats)

            for zonal_stats_name, zonal_stats_value in zonal_stats_result.items():
                shape_feature.SetField(zonal_stats_name,
                                       float(zonal_stats_value))

            lock.acquire()
            shape_layer.SetFeature(shape_feature)
            shape_feature = shape_layer.GetNextFeature()
            lock.release()

    @staticmethod
    def process(shape_dataset, raster_dataset, stats=[]):
        shape_layer = shape_dataset.GetLayer()
        shape_layer.StartTransaction()

        raster_proj = osr.SpatialReference()
        raster_proj.ImportFromWkt(raster_dataset.GetProjection())

        threads = list()

        for band_idx in range(1, raster_dataset.RasterCount + 1):
            for s in stats:
                band_name = "band_{band_idx}_{s}".format(
                    band_idx=str(band_idx),
                    s=s
                )
                try:
                    shape_layer.CreateField(
                        ogr.FieldDefn(band_name, ogr.OFTReal))
                except Exception as e:
                    pass

        for thread_id in range(THREADS):
            x = threading.Thread(target=ZonalStatistics.thread_function,
                                 args=(shape_layer, raster_dataset, stats))
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join()

        shape_layer.CommitTransaction()
        shape_dataset.Destroy()

    @staticmethod
    def zonal_stats(feat, raster, stats=[], nodata_value=None):
        feature_stats = {}
        for band_idx in range(1, raster.RasterCount + 1):
            band_name = "band_{band_idx}".format(band_idx=str(band_idx))
            band = raster.GetRasterBand(band_idx)
            transform = raster.GetGeoTransform()

            if nodata_value:
                nodata_value = float(nodata_value)
                band.SetNoDataValue(nodata_value)

            mem_drv = ogr.GetDriverByName('MEMORY')
            driver = gdal.GetDriverByName('MEM')

            geom = feat.geometry()

            src_offset = ZonalStatistics.bbox_to_pixel_offsets(transform,
                                                               geom.GetEnvelope())
            src_array = band.ReadAsArray(*src_offset)

            if src_array is None:
                continue

            new_gt = (
                (transform[0] + (src_offset[0] * transform[1])),
                transform[1],
                0.0,
                (transform[3] + (src_offset[1] * transform[5])),
                0.0,
                transform[5]
            )

            # Create a temporary vector layer in memory
            mem_ds = mem_drv.CreateDataSource(uuid.uuid4().hex)
            mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
            mem_layer.CreateFeature(feat.Clone())

            # Rasterize it
            rvds = driver.Create(uuid.uuid4().hex, src_offset[2],
                                 src_offset[3], 1, gdal.GDT_Float32)
            rvds.SetGeoTransform(new_gt)
            gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
            rv_array = rvds.ReadAsArray()

            # Mask the source data array with our current feature
            # we take the logical_not to flip 0<->1 to get the correct mask effect
            # we also mask out nodata values explictly
            masked = np.ma.MaskedArray(
                src_array,
                mask=np.logical_or(
                    src_array == nodata_value,
                    np.logical_not(rv_array)
                )
            )

            for s in stats:
                if s in matricial_stats:
                    feature_stats[band_name + "_" + s] = stats_dict[s](masked)
                elif s in vectorial_stats:
                    feature_stats[band_name + "_" + s] = stats_dict[s](geom)
                else:
                    raise Exception("stats {s} not found.".format(s))

        return feature_stats

    @staticmethod
    def bbox_to_pixel_offsets(gt, bbox):
        originX = gt[0]
        originY = gt[3]
        pixel_width = gt[1]
        pixel_height = gt[5]
        x1 = int((bbox[0] - originX) / pixel_width)
        x2 = int((bbox[1] - originX) / pixel_width)

        y1 = int((bbox[3] - originY) / pixel_height)
        y2 = int((bbox[2] - originY) / pixel_height)

        xsize = x2 - x1
        ysize = y2 - y1

        return (x1, y1, xsize, ysize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_raster', type=str, nargs='?',
                        help="Input raster e.g input.tif")
    parser.add_argument('--input_vector', type=str, nargs='?',
                        help="Input vector e.g input.gpkg")
    parser.add_argument('--stats', type=str, nargs='+', default=["max"],
                        help="stats available: max, min, mean, std, sum, count")
    args = parser.parse_args()

    input_raster_filename = args.input_raster
    input_vector_filename = args.input_vector
    stats = args.stats

    input_raster_dataset = gdal.Open(input_raster_filename)
    input_vector_dataset = ogr.Open(input_vector_filename, 1)

    zonal_statistics = ZonalStatistics.process(
        shape_dataset=input_vector_dataset,
        raster_dataset=input_raster_dataset,
        stats=stats)
