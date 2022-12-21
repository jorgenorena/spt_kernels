from scipy import interpolate
import numpy as np

import os
os.chdir("/Users/jorgenorena/Dropbox/Academic_2021/fnl_squeezed_measure/matteo_codes/")
P_list = np.loadtxt('power.dat')
P = interpolate.interp1d(P_list[:,0], P_list[:,1])

def F3(k1, k2, k3, k1dk2, k1dk3, k2dk3):
    return ((k1dk2 + k1dk3)*(k1**2 + 2*k1dk2 + k2**2)*(k1**2 + 2*k1dk3 + k3**2)*(3*k2**2*(k2dk3 + k3**2) + 4*k2dk3*(k2**2 + 2*k2dk3 + k3**2) + 3*k3**2*(k2**2 + k2dk3))*(k1**2 + 2*k1dk2 + 2*k1dk3 + k2**2 + 2*k2dk3 + k3**2) + (k1dk2 + k2dk3)*(k1**2 + 2*k1dk2 + k2**2)*(k2**2 + 2*k2dk3 + k3**2)*(3*k1**2*(k1dk3 + k3**2) + 4*k1dk3*(k1**2 + 2*k1dk3 + k3**2) + 3*k3**2*(k1**2 + k1dk3))*(k1**2 + 2*k1dk2 + 2*k1dk3 + k2**2 + 2*k2dk3 + k3**2) + (k1dk3 + k2dk3)*(k1**2 + 2*k1dk3 + k3**2)*(k2**2 + 2*k2dk3 + k3**2)*(3*k1**2*(k1dk2 + k2**2) + 4*k1dk2*(k1**2 + 2*k1dk2 + k2**2) + 3*k2**2*(k1**2 + k1dk2))*(k1**2 + 2*k1dk2 + 2*k1dk3 + k2**2 + 2*k2dk3 + k3**2) + (7*k1**2*(k1dk2 + k1dk3 + k2**2 + 2*k2dk3 + k3**2) + (k1dk2 + k1dk3)*(k1**2 + 2*k1dk2 + 2*k1dk3 + k2**2 + 2*k2dk3 + k3**2))*(k1**2 + 2*k1dk2 + k2**2)*(k1**2 + 2*k1dk3 + k3**2)*(3*k2**2*(k2dk3 + k3**2) + 4*k2dk3*(k2**2 + 2*k2dk3 + k3**2) + 3*k3**2*(k2**2 + k2dk3)) + (7*k2**2*(k1**2 + k1dk2 + 2*k1dk3 + k2dk3 + k3**2) + (k1dk2 + k2dk3)*(k1**2 + 2*k1dk2 + 2*k1dk3 + k2**2 + 2*k2dk3 + k3**2))*(k1**2 + 2*k1dk2 + k2**2)*(k2**2 + 2*k2dk3 + k3**2)*(3*k1**2*(k1dk3 + k3**2) + 4*k1dk3*(k1**2 + 2*k1dk3 + k3**2) + 3*k3**2*(k1**2 + k1dk3)) + (7*k3**2*(k1**2 + 2*k1dk2 + k1dk3 + k2**2 + k2dk3) + (k1dk3 + k2dk3)*(k1**2 + 2*k1dk2 + 2*k1dk3 + k2**2 + 2*k2dk3 + k3**2))*(k1**2 + 2*k1dk3 + k3**2)*(k2**2 + 2*k2dk3 + k3**2)*(3*k1**2*(k1dk2 + k2**2) + 4*k1dk2*(k1**2 + 2*k1dk2 + k2**2) + 3*k2**2*(k1**2 + k1dk2)) + 7*(k1**2 + k1dk2 + k1dk3)*(k1**2 + 2*k1dk2 + k2**2)*(k1**2 + 2*k1dk3 + k3**2)*(k2**2 + 2*k2dk3 + k3**2)*(5*k2**2*(k2dk3 + k3**2) + 2*k2dk3*(k2**2 + 2*k2dk3 + k3**2) + 5*k3**2*(k2**2 + k2dk3)) + 7*(k1**2 + 2*k1dk2 + k2**2)*(k1**2 + 2*k1dk3 + k3**2)*(k1dk2 + k2**2 + k2dk3)*(k2**2 + 2*k2dk3 + k3**2)*(5*k1**2*(k1dk3 + k3**2) + 2*k1dk3*(k1**2 + 2*k1dk3 + k3**2) + 5*k3**2*(k1**2 + k1dk3)) + 7*(k1**2 + 2*k1dk2 + k2**2)*(k1**2 + 2*k1dk3 + k3**2)*(k1dk3 + k2dk3 + k3**2)*(k2**2 + 2*k2dk3 + k3**2)*(5*k1**2*(k1dk2 + k2**2) + 2*k1dk2*(k1**2 + 2*k1dk2 + k2**2) + 5*k2**2*(k1**2 + k1dk2)))/(756*k1**2*k2**2*k3**2*(k1**2 + 2*k1dk2 + k2**2)*(k1**2 + 2*k1dk3 + k3**2)*(k2**2 + 2*k2dk3 + k3**2))

def F2(k1, k2, k1dk2):
    return k1dk2/(2*k2**2) + 5/7 + 2*k1dk2**2/(7*k1**2*k2**2) + k1dk2/(2*k1**2)

def bispectrum(k1, k2, k3):
   k1dk2 = (k3**2 - k1**2 - k2**2)/2
   k1dk3 = (k2**2 - k1**2 - k3**2)/2
   k2dk3 = (k1**2 - k2**2 - k3**2)/2

   return 2*F2(k1,  k2, k1dk2)*P(k1)*P(k2) + 2*F2(k1, k3, k1dk3)*P(k1)*P(k3) \
      + 2*F2(k2, k3, k2dk3)*P(k2)*P(k3)

def trispectrum(k1, k2, k3, k4, k1pk2, k1pk4):
   k1dk2 = (k1pk2**2 - k1**2 - k2**2)/2
   k1dk4 = (k1pk4**2 - k1**2 - k4**2)/2
   k1dk3 = - k1**2 - k1dk2 - k1dk4  
   k1pk3sq = k1**2 + k3**2 + 2*k1dk3
   k2dk3 = (k1pk4**2 - k2**2 - k3**2)/2
   k2dk4 = (k1pk3sq - k2**2 - k4**2)/2
   k3dk4 = (k1pk2**2 - k3**2 - k4**2)/2

   f3_piece = F3(k2, k3, k4, k2dk3, k2dk4, k3dk4)*P(k2)*P(k3)*P(k4) \
               + F3(k1, k3, k4, k1dk3, k1dk4, k3dk4)*P(k1)*P(k3)*P(k4) \
               + F3(k1, k2, k4, k1dk2, k1dk4, k2dk4)*P(k1)*P(k2)*P(k4) \
               + F3(k1, k2, k3, k1dk2, k1dk3, k2dk3)*P(k1)*P(k2)*P(k3)
   
   f2_piece = F2(k3, k1pk3sq**0.5, - k1dk3 - k3**2)\
               *F2(k4, k1pk3sq**0.5, - k2dk4 - k4**2)\
                     *P(k3)*P(k4)*P(k1pk3sq**0.5)\
            + F2(k4, k1pk4, - k1dk4 - k4**2)\
               *F2(k3, k1pk4, - k2dk3 - k3**2)*P(k3)*P(k4)*P(k1pk4)\
            + F2(k2, k1pk2, - k1dk2 - k2**2)\
               *F2(k4, k1pk2, - k3dk4 - k4**2)*P(k2)*P(k4)*P(k1pk2)\
            + F2(k4, k1pk4, - k1dk4 - k4**2)\
               *F2(k2, k1pk4, - k2dk3 - k2**2)*P(k2)*P(k4)*P(k1pk4)\
            + F2(k2, k1pk2, - k1dk2 - k2**2)\
               *F2(k3, k1pk2, - k3dk4 - k3**2)*P(k2)*P(k3)*P(k1pk2)\
            + F2(k3, k1pk3sq**0.5, - k1dk3 - k3**2)\
               *F2(k2, k1pk3sq**0.5, - k2dk4 - k4**2)\
                     *P(k2)*P(k3)*P(k1pk3sq**0.5)\
            + F2(k1, k1pk2, - k1dk2 - k1**2)\
               *F2(k4, k1pk2, - k3dk4 - k4**2)*P(k1)*P(k4)*P(k1pk2)\
            + F2(k4, k1pk3sq**0.5, - k2dk4 - k4**2)\
               *F2(k1, k1pk3sq**0.5, - k1dk3 - k1**2)\
                     *P(k1)*P(k4)*P(k1pk3sq**0.5)\
            + F2(k1, k1pk2, - k1dk2 - k1**2)\
               *F2(k3, k1pk2, - k3dk4 - k3**2)*P(k1)*P(k3)*P(k1pk2)\
            + F2(k3, k1pk4, - k2dk3 - k3**2)\
               *F2(k1, k1pk4, - k1dk4 - k1**2)*P(k1)*P(k3)*P(k1pk4)\
            + F2(k1, k1pk3sq**0.5, - k1dk3 - k1**2)\
               *F2(k2, k1pk3sq**0.5, - k2dk4 - k2**2)\
                     *P(k1)*P(k2)*P(k1pk3sq**0.5)\
            + F2(k2, k1pk4, - k2dk3 - k2**2)\
               *F2(k1, k1pk4, - k1dk4 - k1**2)*P(k1)*P(k2)*P(k1pk4)

   return 6*f3_piece + 4*f2_piece