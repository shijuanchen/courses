import math

re = 6.378e+6
u = 3.986e+14
J2 = 0.00108263

def cal_angular_speed(hs):
    r = re + hs
    ang_speed = math.sqrt(u/math.pow(r, 3))
    return ang_speed

def cal_period(hs):
    r = re + hs
    ang_speed = math.sqrt(u / math.pow(r, 3))
    period = 2.0 * 3.1415 / float(ang_speed)
    return period

def cal_precession(hs, i):
    rad_i = math.radians(i)
    temp = math.pow((re + hs), -3.5)
    precession = -3.0 / 2.0 * J2 * re * re * math.sqrt(u) * temp * math.cos(rad_i)
    return precession

