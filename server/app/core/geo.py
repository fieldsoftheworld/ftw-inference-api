import math


def calculate_area_km2(bbox: list[float]) -> float:
    """
    Calculate the approximate area in square kilometers of a bounding box in EPSG:4326

    Args:
        bbox: list of 4 coordinates [minX, minY, maxX, maxY] in WGS84 (EPSG:4326)

    Returns:
        Approximate area in square kilometers
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    earth_radius = 6371.0

    # Convert latitude differences to kilometers
    lat_distance = earth_radius * math.radians(max_lat - min_lat)

    # Calculate average latitude for longitude distance calculation
    avg_lat = math.radians((min_lat + max_lat) / 2)

    # Convert longitude differences to kilometers at this latitude
    lon_distance = earth_radius * math.cos(avg_lat) * math.radians(max_lon - min_lon)

    # Calculate area
    area_km2 = lat_distance * lon_distance

    return abs(area_km2)
