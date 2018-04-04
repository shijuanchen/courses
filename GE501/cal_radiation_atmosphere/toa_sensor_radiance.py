# This script calculates the top-of-atmosphere radiance received by the sensor
import math as m

def toa_sensor_radiance(reflectance, toa_irradiance, zenith1, zenith2, opt_thck):
    PI = 3.1415
    zenith1 = m.radians(zenith1)
    zenith2 = m.radians(zenith2)
    radiance = reflectance / PI * toa_irradiance * m.cos(zenith1) * m.exp(-opt_thck/m.cos(zenith1)) * m.exp(-opt_thck/m.cos(zenith2))
    return radiance

L3 = toa_sensor_radiance(0.7, 87.66, 27, 12, 0.12)
print 'L3', L3
L4 = toa_sensor_radiance(0.7, 144.25, 27, 12, 0.08)
print 'L4', L4