# This script calculates the view angle of satellite.
import math as m

def cal_view_angle(H, Bsat, Lsat, B, L):
    R = 6.378140e+6
    Rsat = R + H
    Bsat = m.radians(Bsat)
    Lsat= m.radians(Lsat)
    B = m.radians(B)
    L = m.radians(L)

    cos_view_angle = ( Rsat * ( m.cos(Lsat-L) * m.cos(Bsat) * m.cos(B) + m.sin(Bsat) * m.sin(B) ) - R ) * \
                     (Rsat**2 + R**2 - 2 * R * Rsat * (m.cos(Lsat-L) * m.cos(Bsat) * m.cos(B) + m.sin(Bsat) * m.sin(B) ))**(-0.5)
    print cos_view_angle
    view_angle = m.acos(cos_view_angle)
    view_angle = m.degrees(view_angle)

    numerator = m.sin(Lsat-L) * m.cos(Bsat)
    print 'numerator',numerator
    denominator = m.cos(Lsat-L) * m.cos(Bsat) * m.sin(B) - m.sin(Bsat) * m.cos(B)
    print 'denominator',denominator
    tan_azimuth = numerator/denominator
    print tan_azimuth
    azimuth = m.atan(tan_azimuth)
    azimuth = m.degrees(azimuth)
    return view_angle, azimuth

H = 5e+5
Bsat = 38.5
Lsat = 114.93
B = 39.917
L = 116.417
view_angle, azimuth = cal_view_angle(H, Bsat, Lsat, B, L)
print 'view angle=', view_angle, 'azimuth=', azimuth