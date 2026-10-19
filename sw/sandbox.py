from satellite_tle import fetch_latest_tles
import logging
import time
import traceback
from dataclasses import dataclass

from datetime import datetime, timezone

import numpy as np
import astropy.units as u

from sgp4.api import Satrec, SGP4_ERRORS
from astropy.time import Time
from astropy.coordinates import (
    TEME,
    ITRS,
    EarthLocation,
    AltAz,
    CartesianRepresentation,
)

logger = logging.getLogger(__name__)

TLE_LINE_SIZE: int = 69

LAT_DEG: float = -3.745835525799939
LON_DEG: float = -38.539487445532046
HEIGHT_METERS: int = 0

TLE_EXPIREATION_TIME_SEC: int = 60

@dataclass
class TLEInfo():
    tle1: str
    tle2: str
    acquisition_time: int
    raw_data: dict[int,any]

class SatTracker():
    def __init__(self,norad_id: int):
        self.__norad_id: int = norad_id
        self.last_tle: TLEInfo = None

    def set_norad_id(self,norad_id: int):
        self.__norad_id = norad_id

    def get_norad_id(self) -> int:
        return self.__norad_id

    def update_sat_angles(self,force_tle_update: bool = False):
        if self.__is_last_tle_expired() == True or force_tle_update == True:
            if self.__update_tle() == False:
                logger.info("Failed to update TLE info")
                return

        try:
            self.__update_observer_angles()
        except Exception as e:
            logger.error(f"Failed to update Satellite angles: {str(e)}")

    def __is_last_tle_expired(self) -> bool:
        if self.last_tle is None:
            return True

        if time.time() - self.last_tle.acquisition_time > TLE_EXPIREATION_TIME_SEC:
            return True

        return False

    @staticmethod
    def __parse_tle(tle_info) -> TLEInfo:
        logger.debug(tle_info)

        if not isinstance(tle_info,dict):
            logger.error("TLE Info is not dict...")
            raise Exception("TLE Wrong Format")
        
        tle_data: list[str] = tle_info[25544]
        tle1: str = [tle for tle in tle_data[1] if str(tle).startswith("1 ")][0]
        tle2: str = [tle for tle in tle_data[1] if str(tle).startswith("2 ")][0]

        if len(tle1) != len(tle2) != TLE_LINE_SIZE:
            raise Exception("TLE Data seems wrong")

        return TLEInfo(tle1,tle2,int(time.time()),tle_info)

    def __update_tle(self) -> bool:
        logger.info("Updating TLE Info...")
        try:
            self.last_tle = SatTracker.__parse_tle(fetch_latest_tles([self.__norad_id]))
            logger.info(self.last_tle)

        except Exception as e:
            logger.error(f"Failed to update TLE for ID {self.__norad_id}, exception: {str(e)}")
            logger.error(traceback.print_exc())
            self.last_tle = None
            return False
        
        return True

    def __update_observer_angles(self):
        satellite = Satrec.twoline2rv(self.last_tle.tle1,self.last_tle.tle2)

        observer = EarthLocation.from_geodetic(
            lon=LON_DEG * u.deg,
            lat=LAT_DEG * u.deg,
            height=HEIGHT_METERS * u.m,
        )

        t = Time(datetime.now(timezone.utc), scale="utc")

        error, pos, vel = satellite.sgp4(t.jd1, t.jd2)

        if error != 0:
            raise RuntimeError(SGP4_ERRORS[error])

        # Posicao em TEME, em quilometros
        pos_teme = CartesianRepresentation(np.array(pos) * u.km)

        teme = TEME(pos_teme, obstime=t)

        sat_itrs = teme.transform_to(ITRS(obstime=t))

        # Posicao geocentrica do observador no mesmo
        # referencial e no mesmo instante
        obs_itrs = observer.get_itrs(t)

        # Vetor do observador ate o satelite
        vetor_topocentrico = (sat_itrs.cartesian - obs_itrs.cartesian)

        # Construir o referencial topocentrico
        topo_itrs = ITRS(vetor_topocentrico,obstime=t,location=observer)

        horizontal = topo_itrs.transform_to(
            AltAz(obstime=t,location=observer,pressure=0 * u.hPa,)
        )

        azimute = horizontal.az.deg % 360
        elevacao = horizontal.alt.deg
        distancia_km = horizontal.distance.to_value(u.km)

        logger.info(f"UTC:       {t.isot}")
        logger.info(f"Azimuth:   {azimute:.2f} degrees")
        logger.info(f"Elevation:  {elevacao:.2f} degrees")
        logger.info(f"Distance: {distancia_km:.2f} km")        

def main():
    logging.basicConfig(
    level=logging.INFO,
    format="{asctime}s.{msecs}ms - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    )
    norad_id_iss = 25544 # ISS (ZARYA)
    sat_tracker = SatTracker(norad_id_iss)

    while True:
        sat_tracker.update_sat_angles()
        time.sleep(10)

if __name__ == "__main__":
    main()