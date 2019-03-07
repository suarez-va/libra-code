#*********************************************************************************
#* Copyright (C) 2018 Alexey V. Akimov
#*
#* This file is distributed under the terms of the GNU General Public License
#* as published by the Free Software Foundation, either version 2 of
#* the License, or (at your option) any later version.
#* See the file LICENSE in the root directory of this distribution
#* or <http://www.gnu.org/licenses/>.
#*
#*********************************************************************************/

import sys
import cmath
import math
import os

if sys.platform=="cygwin":
    from cyglibra_core import *
elif sys.platform=="linux" or sys.platform=="linux2":
    from liblibra_core import *

from libra_py import models_LVC, units


def run_NEFGRL_populations(omega_DA, V, omega_nm, gamma_nm, req_nm, shift_NE, method, beta, dyn_type, dtau, tmax, dt, filename):

    
    f = open(filename, "w")
    f.close()

    nsteps = int(tmax/dt)+1
    summ, P, k = 0.0, 1.0, 0.0  # probability of donor state


    for step in xrange(nsteps):

        t = step*dt

        f = open(filename, "a")
        f.write("%8.5f  %8.5f  %8.5f \n" % (t, k, P))
        f.close()
 
        # k = k(t')
        k = NEFGRL_rate(t, omega_DA, V, omega_nm, gamma_nm, req_nm, shift_NE, method, beta, dyn_type, dtau)
        
        summ += k * dt; 
        P = math.exp(-summ)  # exp(- int dt' k(t'))



def run_Test1(omega_DA, V, omega_nm, gamma_nm, req_nm, shift_NE, method, beta, dyn_type, t, dtau, filename):

    f = open(filename, "w")
    f.close()

    nsteps = int(t/dtau)
    nomega = len(omega_nm)

    """
     int_0_t { C(t,tau) dtau }  for a fixed t
    """

    integ = 0.0+0.0j

    for step in xrange(nsteps):
        tau = step*dtau
        argg, lin = 0.0+0.0j, 0.0+0.0j 

        for w in range(nomega):
            if method==0:
                argg = argg + Integrand_NE_exact(t, tau, omega_DA, omega_nm[w], shift_NE[w], req_nm[w], beta)
                lin = lin + Linear_NE_exact(t, tau, gamma_nm[w], omega_nm[w], shift_NE[w], req_nm[w], beta)
            elif method==1:
                argg = argg + Integrand_NE_LSC(t, tau, omega_DA, omega_nm[w], shift_NE[w], req_nm[w], beta)
                lin = lin + Linear_NE_LSC(t, tau, gamma_nm[w], omega_nm[w], shift_NE[w], req_nm[w], beta)
            elif method==2:
                argg = argg + Integrand_NE_CAV(t, tau, omega_DA, omega_nm[w], shift_NE[w], req_nm[w], beta)
                lin = lin + Linear_NE_CAV(t, tau, gamma_nm[w], omega_nm[w], shift_NE[w], req_nm[w], beta)
            elif method==3:
                argg = argg + Integrand_NE_CD(t, tau, omega_DA, omega_nm[w], shift_NE[w], req_nm[w], beta)
                lin = lin + Linear_NE_CD(t, tau, gamma_nm[w], omega_nm[w], shift_NE[w], req_nm[w], beta)
            elif method==4:
                argg = argg + Integrand_NE_W0(t, tau, omega_DA, omega_nm[w], shift_NE[w], req_nm[w], beta)
                lin = lin + Linear_NE_W0(t, tau, gamma_nm[w], omega_nm[w], shift_NE[w], req_nm[w], beta)
            elif method==5:
                argg = argg + Integrand_NE_Marcus(t, tau, omega_DA, omega_nm[w], shift_NE[w], req_nm[w], beta)
                lin = lin + Linear_NE_Marcus(t, tau, gamma_nm[w], omega_nm[w], shift_NE[w], req_nm[w], beta)

        C = 0.0+0.0j
        if dyn_type==0:
            C = cmath.exp(argg) * V * V
        elif dyn_type==1:
            C = cmath.exp(argg) * lin

        integ = integ + C*dtau

        f = open(filename, "a")
        f.write("%8.5f  %8.5f  %8.5f %8.5f  %8.5f %8.5f  %8.5f %8.5f  %8.5f\n" 
                % (tau, argg.real, argg.imag, lin.real, lin.imag, C.real, C.imag, integ.real, integ.imag))
        f.close()



