import math

def cal_exitance_wv(wavelength, temperature):
    C1 = 3.742e-16
    C2 = 1.439e-2
    x = C2/float(wavelength * temperature)
    exitance_wv = C1 * math.pow(wavelength, -5) * math.pow((math.exp(x)-1), -1)
    return exitance_wv

#wavelength = 1e-6
#temperature = 273
#print wavelength, temperature, cal_exitance_wv(wavelength, temperature)
#
# wv1 = 0.95e-6
# wv2 = 0.483e-6
# t1 = 6000
# t2 = 6000
# m1 = cal_exitance_wv(wv1, t1)
# m2 = cal_exitance_wv(wv2, t2)
# ratio = m1/m2
# print 'ratio', ratio

wv = 2.215e-6
t = 6000
print cal_exitance_wv(wv, t)