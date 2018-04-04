import math

def cal_temperature_et(exitance, wavelength):
    C1 = 3.742e-16
    C2 = 1.439e-2
    x = exitance**(-1) * C1 * wavelength**(-5) + 1
    temperature = C2 * wavelength**(-1) * (math.log(x))**(-1)
    return temperature

#exitance = 4.79922600567e-09
#wavelength = 1e-6
#print exitance, wavelength, cal_temperature_et(exitance, wavelength)

et1 = 1.719e7
wv1 = 10e-6
print et1, wv1, cal_temperature_et(et1, wv1)