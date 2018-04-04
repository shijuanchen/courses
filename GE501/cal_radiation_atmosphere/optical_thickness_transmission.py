# This script plot the relationship between Rayleigh optical depth and wavelenght.

import matplotlib.pyplot as plt
import numpy as np

def opt_thck_wvl(wvl):
    opt_thickness = 0.008569 * wvl**(-4) * (1 + 0.0113 * wvl**(-2) + 0.00013 * wvl**(-4))
    return opt_thickness

def plot_opt_thck_wvl():
    wvl1 = np.arange(0.2, 1.0, 0.05)
    wvl2 = np.arange(0.2, 1.0, 0.0001)
    output_path = r'/Users/shijuanchen/Desktop/' \
                  r'Fall 2017/GE 501 Advanced Remote Sensing/homework 4/cal_radiation_atmosphere/'
    output_file_path = output_path + 'opt_thck.txt'
    with open(output_file_path, 'w') as f:
        f.write('wavelength,')
        f.write(','.join(str(x) for x in wvl1))
        f.write('\n')
        f.write('opt_thck,')
        opt_thck1 = opt_thck_wvl(wvl1)
        f.write(','.join('%.4f' % y for y in opt_thck1))
    f.close()

    plt.figure(figsize=(5,5))
    plt.plot(wvl1, opt_thck_wvl(wvl1), 'bo',wvl2, opt_thck_wvl(wvl2),'r-')
    plt.xlabel("wavelength ($\mu m$)")
    plt.ylabel('Optical thickness $\\tau$')
    output_path = r'/Users/shijuanchen/Desktop/' \
                  r'Fall 2017/GE 501 Advanced Remote Sensing/homework 4/cal_radiation_atmosphere/opt_thck.png'
    plt.savefig(output_path)
    plt.close()

def cal_vert_transmission(wvl):
    opt_thickness = opt_thck_wvl(wvl)
    transmission = np.exp(-opt_thickness) # math.exp does not accept array input, so use numpy.exp
    return transmission

def plot_transmission():
    wvl1 = np.arange(0.2, 1.0, 0.05)
    wvl2 = np.arange(0.2, 1.0, 0.0001)
    output_path = r'/Users/shijuanchen/Desktop/' \
                  r'Fall 2017/GE 501 Advanced Remote Sensing/homework 4/cal_radiation_atmosphere/'
    output_file_path = output_path + 'transmission.txt'
    with open(output_file_path, 'w') as f:
        f.write('wavelength,')
        f.write(','.join(str(x) for x in wvl1))
        f.write('\n')
        f.write('transmission,')
        transmission1 = cal_vert_transmission(wvl1)
        f.write(','.join('%.4f' % y for y in transmission1))
    f.close()
    plt.figure(figsize=(5, 5))
    plt.plot(wvl1, transmission1, 'bo', wvl2, cal_vert_transmission(wvl2), 'r-')
    plt.xlabel("wavelength ($\mu m$)")  #see https://matplotlib.org/users/mathtext.html
    plt.ylabel('transmission $Tr$')
    output_plot_path = output_path + 'transmission.png'
    plt.savefig(output_plot_path)
    #plt.show()
    plt.close()

wv_3 = np.array([0.485, 0.56, 0.66, 0.83, 1.65, 2.215])
print wv_3
print opt_thck_wvl(wv_3)
print cal_vert_transmission(wv_3)