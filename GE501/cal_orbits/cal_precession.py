import math

def cal_precession(hs, i):
    rad_i = math.radians(i)
    J2 = 0.00108263
    re = 6.378e+6
    u = 3.986e+14
    temp = math.pow((re + hs), -3.5)
    precession = -3.0 / 2.0 * J2 * re * re * math.sqrt(u) * temp * math.cos(rad_i)
    return precession

hs1 = 4.035e+5
i1 = 35
precession1 = cal_precession(hs1, i1)
print "Precession of problem 3 is", precession1

hs2 = 7.053e+5
i2 = 98.21
precession2 = cal_precession(hs2, i2)
print "Precession of problem 4 is", precession2

