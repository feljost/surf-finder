import cv2
import requests
import numpy as np
import threading


def download_tile(url, headers, channels):
    response = requests.get(url, headers=headers)
    arr = np.asarray(bytearray(response.content), dtype=np.uint8)

    if channels == 3:
        return cv2.imdecode(arr, 1)
    return cv2.imdecode(arr, -1)


# Mercator projection
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
def project_with_scale(lat: float, lon: float, scale: float):
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


def download_image(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    zoom: int,
    url: str,
    headers: dict,
    tile_size: int = 256,
    channels: int = 3,
    resolution_scaling:float = 1.0,
) -> np.ndarray:
    """
    Downloads a map region. Returns an image stored either in BGR or BGRA as a `numpy.ndarray`.

    Parameters
    ----------
    `(lat1, lon1)` - Coordinates (decimal degrees) of the top-left corner of a rectangular area

    `(lat2, lon2)` - Coordinates (decimal degrees) of the bottom-right corner of a rectangular area

    `zoom` - Zoom level

    `url` - Tile URL with {x}, {y} and {z} in place of its coordinate and zoom values

    `headers` - Dictionary of HTTP headers

    `tile_size` - Tile size in pixels

    `channels` - Number of channels in the output image. Use 3 for JPG or PNG tiles and 4 for PNG tiles.
    """

    scale = 1 << zoom

    # Find the pixel coordinates and tile coordinates of the corners
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    img = np.ndarray((img_h, img_w, channels), np.uint8)

    def build_row(row_number: int):
        for j in range(tl_tile_x, br_tile_x + 1):
            tile = download_tile(
                url.format(x=j, y=row_number, z=zoom), headers, channels
            )

            # Find the pixel coordinates of the new tile relative to the image
            tl_rel_x = j * tile_size - tl_pixel_x
            tl_rel_y = row_number * tile_size - tl_pixel_y
            br_rel_x = tl_rel_x + tile_size
            br_rel_y = tl_rel_y + tile_size

            # Define where the tile will be placed on the image
            i_x_l = max(0, tl_rel_x)
            i_x_r = min(img_w + 1, br_rel_x)
            i_y_l = max(0, tl_rel_y)
            i_y_r = min(img_h + 1, br_rel_y)

            # Define how border tiles are cropped
            cr_x_l = max(0, -tl_rel_x)
            cr_x_r = tile_size + min(0, img_w - br_rel_x)
            cr_y_l = max(0, -tl_rel_y)
            cr_y_r = tile_size + min(0, img_h - br_rel_y)

            img[i_y_l:i_y_r, i_x_l:i_x_r] = tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r]

    threads = []
    for i in range(tl_tile_y, br_tile_y + 1):
        thread = threading.Thread(target=build_row, args=[i])
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if resolution_scaling != 1:
        return resize_image(img, resolution_scaling)

    return img


def resize_image(img: np.ndarray, scale_factor: float):
    return cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)


def calculate_box(lat: float, lon:float, side_length_km:float):
    """
    Calculate the corners of a square box around a given latitude and longitude.

    Parameters:
    lat (float): Latitude of the center point.
    lon (float): Longitude of the center point.
    side_length_km (float): Side length of the square box in kilometers.

    Returns:
    tuple: Two tuples representing the top-left and bottom-right corners of the box.
           Each tuple contains two elements: latitude and longitude.
    """
    # Constants for degrees to km conversion
    one_deg_lat_km = 111.32
    one_deg_lon_km = one_deg_lat_km * np.cos(np.radians(lat))
    
    # Calculate adjustments (offsets = side length / 2 in lat lon)
    lat_adj = side_length_km / (2 * one_deg_lat_km)
    lon_adj = side_length_km / (2 * one_deg_lon_km)

    # Calculate corners
    top_left = (lat + lat_adj, lon - lon_adj)
    bottom_right = (lat - lat_adj, lon + lon_adj)

    return top_left, bottom_right