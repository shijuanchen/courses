#--------------------------------------------------------------------------------------------------------
# Script name: xsection
# Author: Shijuan Chen
# Date: 10/20/2017
# Description: this script illustrates:
#   (1) obtain quadrature [ng,xg,wg]
#   (2) obtain leaf normal orientation pdfs [gL,hL]
#   (3) evaluate the G function [G(mu,phi)]
#   (4) evaluate the Gamma_d function [Gamma_d(phip,mup->phi,mu]
#--------------------------------------------------------------------------------------------------------
import numpy as np
import math as m
import matplotlib.pyplot as plt

PI = 3.141592654

def gauss_quad(ng):
    # This function obtains the gauss quadrature of order ng and checks:
    # (1) if the quardrature weights sum to 2.0
    # (2) if integral of f(x)=x from 0 to 1 is equal to 0.5
    # inputs:
    #   ng: quadrature order (ng should be an even number between 0 and 100)
    # outputs:
    #   xg: ordinates
    #   wg: weights

    if not(ng >0 and ng < 100):
        print "Guadrature order should be between 0 and 100!"
        return 0
    if ng % 2 != 0:
        print "Guadrature order should be an even number!"
        return 0
    xg, wg = np.polynomial.legendre.leggauss(ng)
#    print "xg=", xg
#    print 'wg=', wg

    if abs(np.sum(wg) - 2.0) < 0.00001:
        str_succ = "The quardrature weights sum to 2.0."
#        print str_succ
    else:
        return 0
    sum = 0
    for i in range(0, ng):
        if xg[i] > 0 and xg[i] < 1:
            sum = sum + xg[i] * wg[i]
#    print 'When taking quadrature order of', ng,
#    print ',the integral of f(x)=x from 0 to 1 is equal to', sum
    return xg, wg

def example_integral(ng):
    # This function does the integral whose lower and upper bounds are A and B
    # for example integral_A^B dx |x| , A = -1 and B = +1

    xg, wg = gauss_quad(ng)

    # step 1: define conversion factors conv1 and conv2
    upperlimit = 1.0
    lowerlimit = -1.0
    conv1 = (upperlimit - lowerlimit) / 2.0
    conv2 = (upperlimit + lowerlimit) / 2.0

    # step2: do the integral by making sure the ordinates run between the upperlimit and lowerlimit
    sum = 0
    for i in range(0, ng):
        neword = conv1 * xg[i] + conv2
        sum = sum + abs(neword * wg[i])

    # step 3: make sure not to forget to apply the conversion factor 1 again
    sum = sum * conv1
    print 'When taking quadrature order of', ng,
    print ", the integral of f(x)=|x| from -1 to 1 is equal to", sum

def leaf_normal_pdf(ng, model_type):

    # set hL = 1.0
    hL = []
    gL = []
    for i in range(0, ng):
        hL.append(1.0)

    # obtain the leaf normal distribution function
    # and at the same time check if it satisfies the required condition of normalizaion:
    # e.g. [ integral_0^(PI/2) dthetaL gL(thetaL) = 1.0 ]

    # step 1: define limits of integration and the conversion factors
    upperlimit = PI/2.0
    lowerlimit = 0.0
    conv1 = (upperlimit - lowerlimit) / 2.0
    conv2 = (upperlimit + lowerlimit) / 2.0

    # step 2: do the integral by making sure the ordinates run between the upperlimit and the lowerlimit
    xg, wg = gauss_quad(ng)
    sum = 0.0
    for i in range(0, ng):
        neword = conv1 * xg[i] + conv2
        gL_item = leaf_normal_model(neword, model_type)
        gL.append(gL_item)
        sum = sum + gL[i] * wg[i]
    sum = sum * conv1
    if abs(sum - 1.0) < 0.00001:
        str_succ = "\nThe integral of leaf normal of pdf is equal to 1.0."
#        print str_succ
    else:
        return 0
    return gL

def leaf_normal_model(leaf_inclin, model_type):
    if model_type == 'planophile':
        gL_item = (2.0 / PI) * (1.0 + np.cos(2.0 * leaf_inclin))
    elif model_type == 'erectophile':
        gL_item = (2.0 / PI) * (1.0 - np.cos(2.0 * leaf_inclin))
    elif model_type == 'plagiophile':
        gL_item = (2.0 / PI) * (1.0 - np.cos(4.0 * leaf_inclin))
    elif model_type == 'extremophile':
        gL_item = (2.0 / PI) * (1.0 + np.cos(4.0 * leaf_inclin))
    elif model_type == 'uniform':
        gL_item = 2.0 / PI * np.ones(leaf_inclin.size)
    elif model_type == 'spherical':
       gL_item = np.sin(leaf_inclin)
    else:
        print "Model type error!"
        return 0
    return gL_item

def plot_leaf_normal_model():
    # Plot Fig 3. the leaf inclination in degrees vs leaf normail inclinatoin distribution function
    plt.figure(figsize=(10, 10))
    leaf_inclin_dg = np.arange(0, 90, 0.01)
    leaf_inclin_rd = np.radians(leaf_inclin_dg)
    a, = plt.plot(leaf_inclin_dg, leaf_normal_model(leaf_inclin_rd, 'planophile'), 'b-')
    b, = plt.plot(leaf_inclin_dg, leaf_normal_model(leaf_inclin_rd, 'erectophile'), 'g-')
    c, = plt.plot(leaf_inclin_dg, leaf_normal_model(leaf_inclin_rd, 'plagiophile'), 'r-')
    d, = plt.plot(leaf_inclin_dg, leaf_normal_model(leaf_inclin_rd, 'extremophile'), 'c-')
    e, = plt.plot(leaf_inclin_dg, leaf_normal_model(leaf_inclin_rd, 'uniform'), 'm-')
    #plt.plot(leaf_inclin_dg, leaf_normal_model(leaf_inclin_rd, 'spherical'), 'y-')
    plt.xlabel('the leaf inclination in degrees')
    plt.ylabel('leaf normail inclinatoin distribution function')

    plt.legend([a, b, c, d, e], ['planophile', 'erectophile', 'plagiophile', 'extremophile', 'uniform'])
#    output_path = r'/Users/shijuanchen/Desktop/Fall 2017/GE 645 Physical Models of Remote Sensing/xsections/fig3.png'
#    plt.savefig(output_path)
    plt.show()

def G_dir_function(ng, model_type, muprime, phiprime):
    # This function calculates the G_FUNCTION given any direction of photon travel OMEGA_PRIME and the leaf normal pdf
    # Input:
    #   ng: quadrature order
    #   model_type: the model type describes the leaf normal distribution.
    #   hL: pdf of leaf normal azimuth distribution.
    #       Assume that leaves have no preferences on horizontal directions, set hL = 1.0
    #   muprime: cosine of inclination angle of leaf normal (angle to z axis)
    #   phiprime: azimuth angle of leaf normal (angle to x axis)
    # Output:
    #   G_dir_function
    hL = []
    for i in range(0, ng):
        hL.append(1.0)

    gL = leaf_normal_pdf(ng, model_type)

    phiprime = np.radians(phiprime)

    mu_tp = muprime
    sin_tp = np.sqrt(1.0 - muprime * muprime)

    # Define limits of integration and the conversion factor for integration over thetaL (note the tL suffix)
    upperlimit_tL = PI / 2.0
    lowerlimit_tL = 0.0
    conv1_tL = (upperlimit_tL - lowerlimit_tL) / 2.0
    conv2_tL = (upperlimit_tL + lowerlimit_tL) / 2.0

    # Define limits of integratoin and the conversion factors for integration over phiL (note the pL suffix)
    upperlimit_pL = 2.0 * PI
    lowerlimit_pL = 0.0
    conv1_pL = (upperlimit_pL - lowerlimit_pL) / 2.0
    conv2_pL = (upperlimit_pL + lowerlimit_pL) / 2.0

    # Do the integral over theta_L
    sum_tL = 0.0
    xg, wg = gauss_quad(ng)
    for i in range(0, ng):
        neword_tL = conv1_tL * xg[i] + conv2_tL
        mu_tL = np.cos(neword_tL)
        sin_tL = np.sin(neword_tL)

        # Do the integral over phi_L
        sum_pL = 0.0
        for j in range(0, ng):
            neword_pL = conv1_pL * xg[j] + conv2_pL
            dotproduct = np.abs(mu_tL * mu_tp + sin_tL * sin_tp * np.cos(neword_pL - phiprime))
            sum_pL = sum_pL + wg[j] * hL[j] / (2.0 * PI) * dotproduct

        # Finish the phi_L integral
        sum_pL = sum_pL * conv1_pL
        sum_tL = sum_tL + wg[i] * gL[i] * sum_pL

    # Finish the theta_L integral
    sum_tL = sum_tL * conv1_tL
    Gdir = sum_tL
    return Gdir

def plot_Gdir():
    # Plot Fig 3. the leaf inclination in degrees vs leaf normail inclinatoin distribution function
    plt.figure(figsize=(10, 10))
    thetaprime = np.arange(0, 90, 0.01)
    thetaprime_rd = np.radians(thetaprime)
    muprime = np.cos(thetaprime_rd)
    phiprime  = 0 # the plot won't change with phiprime. Giving any value to phiprime is fine.
    #G_dir_function(6, model_type, thetaprime, phiprime)
    ng = 30
    a, = plt.plot(thetaprime, G_dir_function(ng, 'planophile', muprime, phiprime), 'b-')
    b, = plt.plot(thetaprime, G_dir_function(ng, 'erectophile', muprime, phiprime), 'g-')
    c, = plt.plot(thetaprime, G_dir_function(ng, 'plagiophile', muprime, phiprime), 'r-')
    d, = plt.plot(thetaprime, G_dir_function(ng, 'extremophile', muprime, phiprime), 'c-')
    e, = plt.plot(thetaprime, G_dir_function(ng, 'spherical', muprime, phiprime), 'm-')

    plt.xlabel('Projection Polar Angle in Degrees')
    plt.ylabel('Geometry Function')

    plt.legend([a, b, c, d, e], ['planophile', 'erectophile', 'plagiophile', 'extremophile', 'spherical'])
#    output_path = r'/Users/shijuanchen/Desktop/Fall 2017/GE 645 Physical Models of Remote Sensing/xsections/fig8_v2.png'
#    plt.savefig(output_path)
    plt.show()

def G_dif_function(ng):
    # This function checks if the integral over all directions of G function is 0.5, that is:
    # (1/2PI) int_0^2PI dphi^prime int_0^1 dmu^prime G(OMEGA^prime) = 0.5
    # adopted from G_dif_function()
    xg, wg = gauss_quad(ng)

    upperlimit_pp = 2.0 * PI
    lowerlimit_pp = 0.0
    conv1_pp = (upperlimit_pp - lowerlimit_pp) / 2.0
    conv2_pp = (upperlimit_pp + lowerlimit_pp) / 2.0

    G_dif = np.zeros(shape=(ng, ng))
    for i in range(0, ng):
        muprime = xg[i]
        for j in range(0, ng):
            phiprime = conv1_pp * xg[j] + conv2_pp
            phiprime = np.degrees(phiprime)
            G_dif[i][j] = G_dir_function(ng, 'planophile', muprime, phiprime)

    # check for normalization
    # (1/2PI) int_0^2PI dphi^prime int_0^1 dmu^prime G(OMEGA^prime) = 0.5
    sum_tp = 0.0
    for i in range(ng/2, ng):
        sum_pp = 0.0
        for j in range(0, ng):
            sum_pp = sum_pp + wg[j] * G_dif[i][j]
        sum_pp = sum_pp * conv1_pp
        sum_tp = sum_tp + wg[i] * sum_pp
    sum = sum_tp / (2.0 * PI)
    print 'The integral over all directions of G function is ', sum
    return G_dif

def GAMMA_d_function(ng, model_type, muprime, phiprime, mu, phi, rho_Ld, tau_Ld):
    # This function calculates the GAMMA_D function give a direction of photon travel OMEGA^PRIME and the leaf normal pdf

    # set hL = 1.0
    hL = []
    for i in range(0, ng):
        hL.append(1.0)

    gL = leaf_normal_pdf(ng, model_type)

    #phiprime = m.radians(phiprime)

    mu_t = mu
    mu_tp = muprime
    sin_t = m.sqrt(1.0 - mu * mu)
    sin_tp = m.sqrt(1.0 - muprime * muprime)
    xg, wg = gauss_quad(ng)

    # define the limits of integration and the conversion factors for intergration over thetaL (note the tL suffix!)
    upperlimit_tL  = PI / 2.0
    lowerlimit_tL = 0.0
    conv1_tL = (upperlimit_tL - lowerlimit_tL) / 2.0
    conv2_tL = (upperlimit_tL + lowerlimit_tL) / 2.0

    # define the limits of integration and the conversion factors for integration
    upperlimit_pL = 2.0 * PI
    lowerlimit_pL = 0.0
    conv1_pL = (upperlimit_pL - lowerlimit_pL) / 2.0
    conv2_pL = (upperlimit_pL + lowerlimit_pL) / 2.0

    # integral over theta_L
    sum_tL = 0.0
    for i in range(0, ng):
        neword_tL = conv1_tL * xg[i] + conv2_tL
        mu_tL = m.cos(neword_tL)
        sin_tL = m.sin(neword_tL)

        #integral over phi_L
        sum_pL = 0.0
        for j in range(0, ng):
            neword_pL = conv1_pL * xg[j] + conv2_pL
            dotproduct1 = (mu_tL * mu_tp + sin_tL * sin_tp * m.cos(neword_pL - phiprime))
            dotproduct2 = (mu_tL * mu_t + sin_tL * sin_t * m.cos(neword_pL - phi))

            if dotproduct1 * dotproduct2 < 0.0:
                sum_pL = sum_pL + rho_Ld * wg[j] * hL[j] / (2.0 * PI) * abs(dotproduct1 * dotproduct2)
            else:
                sum_pL = sum_pL + tau_Ld * wg[j] * hL[j] / (2.0 * PI) * abs(dotproduct1 * dotproduct2)

        # finish the phi_L integral
        sum_pL = sum_pL * conv1_pL
        sum_tL = sum_tL + wg[i] * gL[i] * sum_pL

    # finish the theta_L integral
    sum_tL = sum_tL * conv1_tL
    Gamma_d = sum_tL
#    print 'Gamma_d', Gamma_d
    return Gamma_d

def GAMMA_d_dir_function(ng, model_type, muprime, phiprime, rho_Ld, tau_Ld):
    # This function evaluates the GAMMA_d function ( (muprime, phiprime) -> (mu, phi))
    # where (mu, phi) are quadrature directions and check for normalization

    upperlimit_p = 2.0 * PI
    lowerlimit_p = 0.0
    conv1_p = (upperlimit_p - lowerlimit_p) / 2.0
    conv2_p = (upperlimit_p + lowerlimit_p) / 2.0

    xg, wg = gauss_quad(ng)

    GAMMA_d_dir = np.zeros(shape=(ng, ng))
    cosine = np.zeros(shape=(ng, ng))
    # get the Gamma_d_dir matrix direction by direction
    for i in range(0, ng):
        mu = xg[i]
        for j in range(0, ng):
            phi = conv1_p * xg[j] + conv2_p
            GAMMA_d_dir[i][j] = GAMMA_d_function(ng, model_type, muprime, phiprime, mu, phi, rho_Ld, tau_Ld)
            theta = m.acos(mu)
            theta_prime = m.acos(muprime)
            #cosine[i][j] = m.sin(phi) * m.sin(phiprime) * m.cos(theta - theta_prime) + m.cos(phi) * m.cos(phiprime)
            cosine[i][j] = (muprime * mu) + (m.sin(m.acos(mu)) * m.sin(theta_prime) * m.cos(phiprime - phi))
    return GAMMA_d_dir, cosine

def CHECK_Gamma_d_dir(ng, model_type, muprime, phiprime, rho_Ld, tau_Ld):
    # check the Gamma_d_dir for normalization
    xg, wg = gauss_quad(ng)
    Gdir = G_dir_function(ng, model_type, muprime, phiprime)
    Gamma_d_dir, cosine = GAMMA_d_dir_function(ng, model_type, muprime, phiprime, rho_Ld, tau_Ld)

    # conversion factors for phiprime
    upperlimit_pp = 2.0 * PI
    lowerlimit_pp = 0.0
    conv1_pp = (upperlimit_pp - lowerlimit_pp) / 2.0
    conv2_pp = (upperlimit_pp + lowerlimit_pp) / 2.0

    # check for normalization

    # (1/PI) int_0^2PI dphi int_0^1 dmu Gamma_d_dir(muprime, phiprime -> mu, phi) = Gdir * (rho + tau_Ld)
    sum_tp = 0.0
    for i in range(0, ng):
        sum_pp = 0.0
        for j in range(0, ng):
            sum_pp = sum_pp + wg[j] * Gamma_d_dir[i][j]
        sum_pp = sum_pp * conv1_pp
        sum_tp = sum_tp + wg[i] * sum_pp
    sum_tp = sum_tp / PI
    sum_tp = sum_tp / (Gdir * (rho_Ld + tau_Ld))
    print 'CHECK_Gamma_d_dir=', sum_tp
    return

#print GAMMA_d_function(6, 'uniform', 0.5, 0, 0.4, 0, 0.01, 0.02)
#print CHECK_Gamma_d_dir(6, 'uniform', 0.9, 0.4, 0.05, 0.1)

def Gamma_d_uniform(cosine_beta, ratio_tw):
    w = 1
    t = ratio_tw
    beta = np.arccos(cosine_beta)
    Gamma_d = w / (3 * PI) * (np.sin(beta) - beta * np.cos(beta)) + t / PI * np.cos(beta)
    return Gamma_d

def plot_Gamma_d_function_uniform():
    plt.figure(figsize=(10,10))
    cosine_beta = np.arange(-1.0, 1.0, 0.05)
    ratio_tw_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    symbol_list = ['b-', 'b--', 'b.-', 'bs-', 'bo-', 'b^-']
    p_list = []
    label_list = []
    for k in range(0, 6):
        p, = plt.plot(cosine_beta, Gamma_d_uniform(cosine_beta, ratio_tw_list[k]), symbol_list[k])
        p_list.append(p)
        label = 't/w = ' + str(ratio_tw_list[k])
        label_list.append(label)
    plt.legend(p_list, label_list)
    plt.xlabel('Cosine of Scattering Angle')
    plt.ylabel('Area Scattering Phase Function')
#    output_path = r'/Users/shijuanchen/Desktop/Fall 2017/GE 645 Physical Models of Remote Sensing/xsections/fig9.png'
#    plt.savefig(output_path)
    plt.show()

def GAMMA_d_dif_function(ng, rho_Ld, tau_Ld):
    # this function evaluates the Gamma_d_dif function for scattering from all quadrature directions to all exit quadrature directions
    upperlimit_pp = 2.0 * PI
    lowerlimit_pp = 0.0
    conv1_pp = (upperlimit_pp - lowerlimit_pp) / 2.0
    conv2_pp = (upperlimit_pp + lowerlimit_pp) / 2.0

    # get the Gamma_d_dif matrix direction by direction
    xg, wg = gauss_quad(ng)

    for i in range(0, ng):
        muprime = xg[i]
        for j in range(0, ng):
            phiprime = conv1_pp * xg[j] + conv2_pp
            CHECK_Gamma_d_dir(ng, 'planophile', muprime, phiprime, rho_Ld, tau_Ld)



def normalized_scattering_phase():
    # normalized scattering phase function, this function has problem
    ng = 40
    model_type = 'planophile'
    muprime = -0.98 # cos(170)
    phiprime = 0
    rho_Ld = 0.5
    tau_Ld = 0.5
    GAMMA_d_dir, cosine = GAMMA_d_dir_function(ng, model_type, muprime, phiprime, rho_Ld, tau_Ld)

    G_dir = G_dir_function(ng, model_type, muprime, phiprime)
    P_d =  2 * GAMMA_d_dir / (rho_Ld * G_dir )
    plt.ylim(0.0, 2.0)
    plt.xlim(-1.0, 1.0)
    yrange = np.arange(0.0, 2.2, 0.2)
    plt.yticks(yrange)
    xrange = np.arange(-1.0, 1.2, 0.2)
    plt.xticks(xrange)
    plt.plot(cosine, P_d, 'b.')
    plt.xlabel('Cosine of Scattering Angle')
    plt.ylabel('Normalized Scattering Phase Function')
#    output_path = r'/Users/shijuanchen/Desktop/Fall 2017/GE 645 Physical Models of Remote Sensing/xsections/fig11.png'
#    plt.savefig(output_path)
    plt.show()

# run the functions:
def main():
    plot_leaf_normal_model()

    plot_Gdir()

    G_dif_function(ng=12)

    print GAMMA_d_function(ng=10, model_type='planophile', muprime=-0.985, phiprime=0.921, mu=0.731, phi=0.276, rho_Ld=0.07, tau_Ld=0.04)

    print GAMMA_d_function(ng=10, model_type='planophile', muprime=0.731, phiprime=0.276, mu=-0.985, phi=0.921, rho_Ld=0.07, tau_Ld=0.04)

    print GAMMA_d_function(ng=8, model_type='planophile', muprime=0, phiprime=0.5, mu=0.8, phi=0.3, rho_Ld=0.5, tau_Ld=0.5)

    print GAMMA_d_function(ng=8, model_type='planophile', muprime=0.8, phiprime=0.3, mu=0, phi=0.5, rho_Ld=0.5, tau_Ld=0.5)

    CHECK_Gamma_d_dir(ng=6, model_type='planophile', muprime=-0.985, phiprime=0.921, rho_Ld=0.07, tau_Ld=0.04)

    GAMMA_d_dif_function(ng=6, rho_Ld=0.07, tau_Ld=0.04)

    plot_Gamma_d_function_uniform()

    normalized_scattering_phase()

main()
