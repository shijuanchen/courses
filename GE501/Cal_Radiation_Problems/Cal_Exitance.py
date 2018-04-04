# Calculate Exitance
import math

def cal_exitance(temperature):
    SIGMA = 5.669e-8
    exitance = SIGMA * math.pow(temperature, 4)
    return exitance

k = [250, 273, 300, 1000, 3200, 6000]

for i in k:
    print i, cal_exitance(i)

k2 = 966
print k2, cal_exitance(k2)