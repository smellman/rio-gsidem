from __future__ import division
import numpy as np


def data_to_rgb(data, nodata=-9999):
    """
    Given an arbitrary (rows x cols) ndarray,
    encode the data into uint8 RGB from an arbitrary
    base and interval. nodata values are represented
    as a fixed RGB value (128, 0, 0).

    Parameters
    -----------
    data: ndarray
        (rows x cols) ndarray of data to encode
    nodata: float or int, optional
        Value representing nodata in the input array. Default is -9999.

    Returns
    --------
    ndarray: rgb data
        a uint8 (3 x rows x cols) ndarray with the
        data encoded. Nodata pixels are set to (128, 0, 0).
    """
    # Replace nodata values with NaN
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    data = data.astype(np.float64)

    rows, cols = data.shape

    # Compute the data range, ignoring NaN
    datarange = np.nanmax(data) - np.nanmin(data)

    # Range check
    if datarange > 256 ** 3:
        raise ValueError("Data range of {} is larger than 256 ** 3".format(datarange))

    # Initialize RGB array
    rgb = np.zeros((3, rows, cols), dtype=np.uint8)

    # Handle NaN (nodata) pixels
    nan_mask = np.isnan(data)  # Mask for NaN values
    valid_mask = ~nan_mask  # Mask for valid values

    # Scale data to fit into 24 bits
    u = 0.01
    x = np.zeros_like(data, dtype=np.float64)  # Initialize x with zeros
    x[valid_mask] = data[valid_mask] / u  # Convert valid data to encoded range

    # Handle overflow and invalid values
    x = np.nan_to_num(x, nan=0, posinf=2 ** 24 - 1, neginf=0)  # Replace invalid values
    x[x < 0] = 0  # Clamp negative values to 0
    x[x > 2 ** 24 - 1] = 2 ** 24 - 1  # Clamp overflow values to max valid value

    # Initialize r, g, b
    r = np.zeros_like(data, dtype=np.uint32)
    g = np.zeros_like(data, dtype=np.uint32)
    b = np.zeros_like(data, dtype=np.uint32)

    # Decode RGB components for valid pixels only
    if np.any(valid_mask):
        r[valid_mask] = (x[valid_mask] // 2 ** 16).astype(np.uint32)
        x[valid_mask] -= r[valid_mask] * 2 ** 16
        g[valid_mask] = (x[valid_mask] // 2 ** 8).astype(np.uint32)
        x[valid_mask] -= g[valid_mask] * 2 ** 8
        b[valid_mask] = x[valid_mask].astype(np.uint32)

    # Assign to RGB channels
    rgb[0] = r.astype(np.uint8)
    rgb[1] = g.astype(np.uint8)
    rgb[2] = b.astype(np.uint8)

    # Set RGB values for nodata pixels
    rgb[0][nan_mask] = 128
    rgb[1][nan_mask] = 0
    rgb[2][nan_mask] = 0

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

    # Initialize the output data array
    data = np.zeros((rows, cols), dtype=np.float64)

    # Decode the RGB into data
    u = 0.01
    data = (rgb[0] * 2 ** 16 + rgb[1] * 2 ** 8 + rgb[2]) * u

    # Handle special case for (128, 0, 0)
    nodata_mask = (rgb[0] == 128) & (rgb[1] == 0) & (rgb[2] == 0)
    data[nodata_mask] = -9999

    # Handle potential overflow values
    overflow_mask = data > 2 ** 23
    data[overflow_mask] = (data[overflow_mask] - (2 ** 24)) * u

    return data

def _range_check(datarange):
    """
    Utility to check if data range is outside of precision for 3 digit base 256
    """
    maxrange = 256 ** 3

    return datarange > maxrange
