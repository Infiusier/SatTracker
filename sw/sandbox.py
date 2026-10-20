import logging
import time
from sat_tracker import *

LAT_DEG: float = -3.745835525799939
LON_DEG: float = -38.539487445532046
HEIGHT_METERS: int = 0

def main():
    logging.basicConfig(
    level=logging.INFO,
    format="{asctime}s.{msecs}ms - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    )
    norad_id_iss = 25544 # ISS (ZARYA)
    sat_tracker = SatTracker(norad_id_iss,LAT_DEG,LON_DEG)

    while True:
        elev,azim = sat_tracker.update_sat_angles()
        sat_tracker.set_tracker_angles(elev,azim)
        time.sleep(1)

if __name__ == "__main__":
    main()