"""Band definitions shared by the whole pipeline.

Order matters: every array in the project stores bands in exactly this order.
"""

GUIDE_BANDS = ["B02", "B03", "B04", "B08"]          # 10 m, used as guidance
TARGET_BANDS = ["B05", "B06", "B07", "B8A", "B11", "B12"]  # 20 m, super-resolved
RED_EDGE_BANDS = ["B05", "B06", "B07", "B8A"]       # bands with a VENuS reference
SWIR_BANDS = ["B11", "B12"]

N_GUIDE = len(GUIDE_BANDS)
N_TARGET = len(TARGET_BANDS)

# Sentinel-2 L2A scene classification (SCL) classes.
SCL_NO_DATA = 0
SCL_SATURATED = 1
SCL_CLOUD_SHADOW = 3
SCL_VEGETATION = 4
SCL_CLOUD_MEDIUM = 8
SCL_CLOUD_HIGH = 9
SCL_CIRRUS = 10
SCL_SNOW = 11
SCL_INVALID = (SCL_NO_DATA, SCL_SATURATED, SCL_CLOUD_SHADOW, SCL_CLOUD_MEDIUM,
               SCL_CLOUD_HIGH, SCL_CIRRUS, SCL_SNOW)


def target_index(band: str) -> int:
    return TARGET_BANDS.index(band)
