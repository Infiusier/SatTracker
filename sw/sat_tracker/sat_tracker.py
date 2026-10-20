from satellite_tle import fetch_latest_tles
import logging
import time
import traceback
from dataclasses import dataclass
import socket

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
TLE_EXPIRATION_TIME_SEC: int = 60

@dataclass
class TLEInfo():
    tle1: str
    tle2: str
    acquisition_time: int
    raw_data: dict[int,any]

class ROTCTLDSocket():
    def __init__(self,host: str = "localhost",port: int = 4533):
        self.host: str = host
        self.port: int = port

    def set_pos(self,elevation_deg: float, azimuth_deg: float, timeout: int = 5) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout) as sock:
                payload: str = f"P {azimuth_deg:.1f} {elevation_deg:.1f}\n"
                logger.info(f"ROTCTL -> {payload}")

                sock.sendall(payload.encode())
                response = sock.recv(1024).decode()
                
                logger.info(response)
                return True
            
        except Exception as e:
            logger.error(f"Fail to send command to rotor: {e}")
            return False

class SatTracker():
    def __init__(self,norad_id: int,lat_deg: float, lon_deg: float, height_m:int = 0, tle_expiration_sec: int = TLE_EXPIRATION_TIME_SEC):
        self.norad_id: int = norad_id
        self.tle_expiration_sec: int = tle_expiration_sec
        self.lat_deg: float = lat_deg
        self.lon_deg: float = lon_deg
        self.height_m: int = height_m
        self.elevation: float = None
        self.azimuth: float = None
        self.__last_tle: TLEInfo = None
        self.__socket: ROTCTLDSocket = ROTCTLDSocket()

    def update_sat_angles(self,force_tle_update: bool = False) -> tuple[float,float]:
        if self.__is_last_tle_expired() == True or force_tle_update == True:
            if self.__update_tle() == False:
                logger.info("Failed to update TLE info")
                return

        try:
            self.__update_observer_angles()
        except Exception as e:
            logger.error(f"Failed to update Satellite angles: {str(e)}")

        return self.elevation,self.azimuth

    def set_tracker_angles(self,elevation_deg: float, azimuth_deg: float):
        if self.__socket.set_pos(elevation_deg,azimuth_deg) == False:
            logger.error("Failed to Update tracker position...")

    def __is_last_tle_expired(self) -> bool:
        if self.__last_tle is None:
            return True

        if time.time() - self.__last_tle.acquisition_time > TLE_EXPIRATION_TIME_SEC:
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
            self.__last_tle = SatTracker.__parse_tle(fetch_latest_tles([self.norad_id]))
            logger.info(self.__last_tle)

        except Exception as e:
            logger.error(f"Failed to update TLE for ID {self.norad_id}, exception: {str(e)}")
            logger.error(traceback.print_exc())
            self.__last_tle = None
            return False
        
        return True

    def __update_observer_angles(self):
        satellite = Satrec.twoline2rv(self.__last_tle.tle1,self.__last_tle.tle2)

        observer = EarthLocation.from_geodetic(
            lon=self.lon_deg * u.deg,
            lat=self.lat_deg * u.deg,
            height=self.height_m * u.m,
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

        self.elevation = elevacao
        self.azimuth = azimute