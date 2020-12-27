import argparse

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, ogr
from skimage import measure, segmentation, exposure
from skimage.filters import rank
from skimage.filters import sobel
from skimage.morphology import disk
from skimage.util import img_as_float
from sklearn import decomposition

ogr.UseExceptions()

DEBUG = False


class Segmentation(object):
    @staticmethod
    def process(raster_dataset, scale_level=0.9, smoth=True):
        num_bands = raster_dataset.RasterCount

        # redução de ruidos
        bands = []
        for index in range(1, num_bands + 1):
            band = raster_dataset.GetRasterBand(index).ReadAsArray()
            if smoth:
                band = rank.median(band, disk(2))
            bands.append(band)

        image_raw = img_as_float(np.dstack(bands))

        # aplicacao do PCA
        image_flat = image_raw.reshape(-1, num_bands)

        pca = decomposition.PCA(n_components=num_bands)
        pca.fit(image_flat)
        image_flat_pca = pca.transform(image_flat)
        image_pca = image_flat_pca.reshape(image_raw.shape)

        image_pca = image_pca.transpose((-1, 0, 1))

        # geração dos gradientes
        gradients = []
        for band in image_pca[:2]:
            gradients.append(sobel(band))

        # MAX/SUM dos gradientes
        gradient = np.sum(gradients, axis=0)

        if DEBUG:
            plt.imshow(gradient, cmap=plt.cm.nipy_spectral,
                       interpolation='nearest')
            plt.show()

        # calculo do histograma cumulativo relativo
        cdf, bins = exposure.cumulative_distribution(gradient)

        # atualizar o valor do gradiente a partir do Nível de Escala
        cdf_values = bins[cdf < scale_level]
        img1 = gradient.copy()
        gradient[gradient <= cdf_values[-1]] = cdf_values[-1]
        img2 = gradient

        if DEBUG:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8),
                                     sharex=True,
                                     sharey=True)
            ax = axes.ravel()

            ax[0].imshow(img1, cmap=plt.cm.nipy_spectral,
                         interpolation='nearest')
            ax[0].set_title("Gradient")

            ax[1].imshow(img2, cmap=plt.cm.nipy_spectral,
                         interpolation='nearest')
            ax[1].set_title("Gradient with Scale Level")

            plt.show()

        local_otsu = rank.otsu(gradient, disk(1))
        gradient_otsu = gradient >= (scale_level * local_otsu)
        gradient_otsu = rank.median(gradient_otsu, disk(1))

        centroids = np.full(gradient_otsu.shape, 0)
        for i in measure.regionprops(measure.label(gradient_otsu)):
            centroids[int(i.centroid[0])][int(i.centroid[1])] = 1

        markers = measure.label(centroids)

        labels = segmentation.watershed(gradient, markers)

        if DEBUG:
            # display results
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 8),
                                     sharex=True,
                                     sharey=True)
            ax = axes.ravel()

            ax[0].imshow(image_raw, cmap=plt.cm.gray, interpolation='nearest')
            ax[0].set_title("Original")

            ax[1].imshow(image_pca[0], cmap=plt.cm.gray,
                         interpolation='nearest')
            ax[1].set_title("PCA - 1")

            ax[2].imshow(gradient, cmap=plt.cm.nipy_spectral,
                         interpolation='nearest')
            ax[2].set_title("Sobel")

            ax[3].imshow(gradient_otsu, cmap=plt.cm.nipy_spectral,
                         interpolation='nearest')
            ax[3].set_title("Otsu")

            ax[4].imshow(markers, cmap=plt.cm.nipy_spectral,
                         interpolation='nearest')
            ax[4].set_title("Markers")

            ax[5].imshow(labels, cmap=plt.cm.nipy_spectral,
                         interpolation='nearest',
                         alpha=.7)
            ax[5].set_title("Segmented")

            plt.show()

        return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?',
                        help="Input image e.g mosaic.tif")
    parser.add_argument('--output', type=str, nargs='?',
                        help="Output segmentation e.g output.shp")
    parser.add_argument('--scale_level', type=float, nargs='?', default=0.9,
                        help="values between 0-1. 0=more targeted  1=little targeted")
    parser.add_argument('--smoth', type=bool, default=False,
                        help="Smooth the image before segmentation")
    args = parser.parse_args()

    input_filename = args.input
    output_filename = args.output
    scale_level = args.scale_level
    smoth = args.smoth

    raster_dataset = gdal.Open(input_filename)

    labels = Segmentation.process(raster_dataset=raster_dataset,
                                  scale_level=scale_level,
                                  smoth=smoth)

    driver = raster_dataset.GetDriver()

    output_dataset = driver.Create(output_filename,
                                   raster_dataset.RasterXSize,
                                   raster_dataset.RasterYSize, 1,
                                   gdal.GDT_Float32)
    output_dataset.SetProjection(raster_dataset.GetProjectionRef())
    output_dataset.SetGeoTransform(raster_dataset.GetGeoTransform())
    output_dataset.GetRasterBand(1).WriteArray(labels)
    output_dataset.FlushCache()  # Write to disk.
