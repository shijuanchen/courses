# This script calcuates the top of atmosphere irradiance.
import math

def toa_irradiance(center_wvl, bandwidth, solar_temperature):
    PI = 3.1415

    # Use the Planck equation to find the spectral exitance of the sun's surface at the center wavelength.
    spectral_exitance = cal_exitance_wv(center_wvl, solar_temperature)

    # Multiply by the bandiwdth to give full-band exitance.
    full_band_exitance = spectral_exitance * bandwidth

    # Find the surface area of the sun
    r = 6.957e+8
    surf_area = 4.0 * PI * r**2

    # Multiply that by the full-band-exitance to give the total radiation flux emitted by the sun
    # into all directions in that band
    total_radiation_flux = surf_area * full_band_exitance

    # Divid the emitted flux by the 4PIE steradians in a sphere
    # to give the radiant intensity of the sun as a point source.
    radiant_intensity = total_radiation_flux / (4.0 * PI)

    # Find how many steradians are associated with one-meter square area situated at the earth-sun distance of 1 au.
    R = 1.496e+11
    omega = 1 / (R**2)

    # Multiply this value by the radiant intensityof the sun to give watts falling on the one-square meter area,
    # which will be the top-of-atmosphere irradiance in this band
    irradiance = omega * radiant_intensity

    return irradiance

def cal_exitance_wv(wavelength, temperature):
    C1 = 3.742e-16
    C2 = 1.439e-2
    x = C2 / float(wavelength * temperature)  # two array multiplication causes problems. Use loop, don't use array.
    exitance_wv = C1 * math.pow(wavelength, -5) * math.pow((math.exp(x) - 1), -1)
    return exitance_wv

def write_toa_irradiance():
    center_wvl = [0.485e-6, 0.56e-6, 0.66e-6, 0.83e-6, 1.65e-6, 2.215e-6]
    bandwidth = [0.07e-6, 0.08e-6, 0.06e-6, 0.14e-6, 0.2e-6, 0.27e-6]
    solar_temperature = [5950, 5820, 5720, 5700, 6000, 6000]
    toa_irrad = []
    for i in range(0, 6):
        toa_irrad.append(toa_irradiance(center_wvl[i], bandwidth[i], solar_temperature[i]))
    print toa_irrad

    output_file_path = r'/Users/shijuanchen/Desktop/Fall 2017/GE 501 Advanced Remote Sensing' \
                       r'/homework 4/cal_radiation_atmosphere/toa_irradiance.txt'
    with open(output_file_path, 'w') as f:
        f.write('center_wvl,')
        center_wvl2 = [str(x) for x in center_wvl]
        f.write(','.join(center_wvl2))
        f.write('\nbandwidth,')
        f.write(','.join(str(x) for x in bandwidth))
        f.write('\nsolar_temperature,')
        f.write(','.join(str(x) for x in solar_temperature))
        f.write('\ntoa_irradiance,')
        f.write(','.join('%.2f' % x for x in toa_irrad))
        f.write('\n')
    f.close()

write_toa_irradiance()