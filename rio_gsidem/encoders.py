from __future__ import division
import numpy as np


def data_to_rgb(data):
    """
    Given an arbitrary (rows x cols) ndarray,
    encode the data into uint8 RGB from an arbitrary
    base and interval

    Parameters
    -----------
    data: ndarray
        (rows x cols) ndarray of data to encode

    Returns
    --------
    ndarray: rgb data
        a uint8 (3 x rows x cols) ndarray with the
        data encoded
    """
    data = data.astype(np.float64)

    rows, cols = data.shape

    datarange = data.max() - data.min()

    if _range_check(datarange):
        raise ValueError("Data of {} larger than 256 ** 3".format(datarange))

    rgb = np.zeros((3, rows, cols), dtype=np.uint8)

    u = 0.01
    x = data / u
    x = np.where(x > 2 ** 23, x / u + (2 ** 24), x)
    r = (x // 2 ** 16).astype(np.uint32)
    x -= r * 2 ** 16
    g = (x // 2 ** 8).astype(np.uint32)
    x -= g * 2 ** 8
    b = x.astype(np.uint32)
    rgb[0] = r.astype(np.uint8)
    rgb[1] = g.astype(np.uint8)
    rgb[2] = b.astype(np.uint8)

    return rgb

def _decode(rgb):
    """
    Given a uint8 (3 x rows x cols) ndarray,
    decode the data into an arbitrary base and interval

    Parameters
    -----------
    rgb: ndarray
        uint8 (3 x rows x cols) ndarray of data to decode

    Returns
    --------
    ndarray: data
        a (rows x cols) ndarray with the data decoded
    """
    rows, cols = rgb.shape[1:]

    data = np.zeros((rows, cols), dtype=np.float64)

    # data += rgb[0] * 256
    # data += rgb[1]
    # data += rgb[2] / 256

    # data -= 32768
    u = 0.01
    data = (rgb[0] * 2 ** 16 + rgb[1] * 2 ** 8 + rgb[2]) * u
    if data > 2 ** 23:
        data = (data - (2 ** 24)) * u

    return data

def _range_check(datarange):
    """
    Utility to check if data range is outside of precision for 3 digit base 256
    """
    maxrange = 256 ** 3

    return datarange > maxrange
