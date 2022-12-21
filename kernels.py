# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 08:59:17 2021

Kernels are in the Scoccimarro convention of dividing each term in the 
\delta expansion by 1/n!.

@author: jorgenorena
"""

import vectors as vect
from vectors import Vector, VectSum, VectScalMul, Dot, code
from vectors import Scalar, ScalPow, ScalMul, ScalSum
from vectors import fract, sympy, translate_sympy, vectors, scalars
from itertools import permutations
from scipy.special import factorial
import sympy as sp

def alpha(k1, k2):
    return vect.Dot((k1 + k2), k1)/vect.Dot(k1,k1)

def beta(k1, k2):
    return (vect.Dot(k1 + k2, k1 + k2)*vect.Dot(k1, k2))/vect.Dot(k1,k1)/vect.Dot(k2,k2)/2

def G(n, ks):
    if n == 1:
        return 1
    else:
        return sum(Gs(m, ks[:m])*\
            (3*alpha(sum(ks[:m]),sum(ks[m:]))*Fs(n-m, ks[m:]) +\
            2*n*beta(sum(ks[:m]),sum(ks[m:]))*Gs(n-m, ks[m:]))/(2*n+3)/(n-1)
            for m in range(1, n))
            
def F(n, ks):
    if n == 1:
        return 1
    else:
        return sum(Gs(m, ks[:m])*\
            ((2*n+1)*alpha(sum(ks[:m]),sum(ks[m:]))*Fs(n-m, ks[m:]) +\
            2*beta(sum(ks[:m]),sum(ks[m:]))*Gs(n-m, ks[m:]))/(2*n+3)/(n-1)
            for m in range(1, n))
            
def Fs(n, ks):
    return vect.ScalPow(int(factorial(n)),-1)*\
        sum(map(lambda x: F(n, x), permutations(ks)))
    
def Gs(n, ks):
    return vect.ScalPow(int(factorial(n)),-1)\
        *sum(map(lambda x: G(n, x), permutations(ks)))

def F2(k1, k2):
    return Fs(2, [k1, k2])


# %%
if __name__=='__main__':
    k1, k2, k3 = vect.vectors('k1', 'k2', 'k3')
    ks = [k1, k2, k3]
    
    vect_kernel = Fs(3, ks)
    sp_kernel = vect.sympy(vect_kernel, "code").simplify()
    print(sp.latex(sp_kernel))

# %%
    sd = list(vect.symbol_dict.keys())
    k1s, k2s, k3s = sp.symbols('k1 k2 k3')
    k1dk2, k1dk3, k2dk3 = sp.symbols('k1dk2 k1dk3 k2dk3')
    def find(iterable, name):
        return next((x for x in iterable if str(x) == name), None)
    replacements = [(find(sd, "k1dk1"), k1s**2),
                    (find(sd, "k2dk2"), k2s**2),
                    (find(sd, "k3dk3"), k3s**2),
                    (find(sd, "k2dk1"), k1dk2),
                    (find(sd, "k3dk1"), k1dk3),
                    (find(sd, "k3dk2"), k2dk3)]
    code_kernel = sp_kernel.subs(replacements).simplify()
    print(code_kernel)
# %%
    ks = [k1, k2]

    vect_kernel = Fs(2, ks)
    sp_kernel = vect.sympy(vect_kernel, "code").simplify()
    print(sp.latex(sp_kernel))
# %%
    code_kernel = sp_kernel.subs(replacements).simplify()
    print(code_kernel)
# %%
