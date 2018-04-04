import cal_orbits

solution1_angular_speed = cal_orbits.cal_angular_speed(8.33e+5)
print solution1_angular_speed

solution1_period = cal_orbits.cal_period(8.33e+5)
print solution1_period

hs1 = 4.035e+5
i1 = 35
precession1 = cal_orbits.cal_precession(hs1, i1)
print "Precession of problem 3 is", precession1

hs2 = 7.053e+5
i2 = 98.21
precession2 = cal_orbits.cal_precession(hs2, i2)
print "Precession of problem 4 is", precession2
