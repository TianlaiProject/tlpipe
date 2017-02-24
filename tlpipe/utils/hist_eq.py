import numpy as np


def hist_eq(img):
    """Implementation of Histogram equalization.

    Histogram equalization is a method in image processing of contrast
    adjustment using the image's histogram. More details see
    https://en.wikipedia.org/wiki/Histogram_equalization
    """

    if not (img.min() >= 0 and img.max() <= 256):
        img = np.around(256.0 * img / (img.max() - img.min())).astype('uint8')
    else:
        img = np.around(img).astype('uint8')

    hist, bins = np.histogram(img.flatten(), 256, [0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]

    return img2


if __name__ == '__main__':
    a = np.array([51, 55, 61, 66, 70, 61, 66, 70, 62, 60, 54, 90, 108, 85, 67, 71, 63, 65, 66, 110, 140, 104, 63, 72, 64, 70, 70, 120, 152, 106, 71, 69, 67, 75, 68, 106, 124, 88, 68, 68, 68, 80, 60, 72, 77, 66, 58, 75, 69, 85, 64, 58, 55, 61, 65, 83, 70, 90, 69, 68, 65, 72, 78, 90]).reshape(8, 8)
    # a = np.random.rand(8, 8) * 256 * 2
    b = hist_eq(a)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(121)
    # plt.imshow(a, aspect='equal', origin='lower', cmap='gray', interpolation='nearest')
    plt.imshow(a, aspect='equal', cmap='gray', interpolation='nearest')
    plt.subplot(122)
    # plt.imshow(b, aspect='equal', origin='lower', cmap='gray', interpolation='nearest')
    plt.imshow(b, aspect='equal', cmap='gray', interpolation='nearest')
    plt.savefig('hist_eq.png')
    plt.close()