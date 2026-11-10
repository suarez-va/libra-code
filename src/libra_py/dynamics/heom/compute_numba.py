# *********************************************************************************
# * Copyright (C) 2026 Victor A. Suarez, Alexey V. Akimov
# *
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 3 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# ***********************************************************************************
"""
.. module:: compute_numba
   :platform: Unix, Windows
   :synopsis: A Numba implementation of the HEOM integrator. It solves the same equations,
       takes the same parameters, and writes the same output as `compute.run_dynamics`
       (the "cpp" backend), but keeps the whole hierarchy in a single numpy array and evaluates
       the derivatives with compiled, thread-parallel kernels.

       Why it is faster than the "cpp" backend:

           * the derivatives of all of the auxiliary density matrices (ADMs) are evaluated on
             flat arrays, without allocating a CMATRIX object for every ADM and every
             intermediate result of every ADM
           * the topology of the hierarchy and the coefficients of the equations are converted
             to arrays only once, instead of being extracted from the Python dictionary of
             parameters (and copied) on every evaluation of the derivatives
           * the RK4 integrator works on 4 preallocated buffers, with no allocation per step
           * whenever the set of nonzero ADMs does not change, the derivatives computed for the
             filtering are reused as the first RK4 stage, which saves one of the 5 evaluations
             of the derivatives per step that the "cpp" backend does when filtering every step
           * the hierarchy is converted from the scaled to the raw representation only on the
             steps on which something is saved, and only the ADMs that are saved are converted

       It is normally used via `compute.run_dynamics` with dyn_params["backend"] = "numba", but
       `run_dynamics` of this module can also be called directly.

       List of functions:
           * build_topology(d, LL, verbosity)
           * compute_truncation_prefactor(params, gamma_matsubara, c_matsubara, gamma_matsubara_ext, c_matsubara_ext)
           * setup_heom(params, Ham)
           * compute_derivatives(heom, rho, out, nonzero, active)
           * scale_adms(heom, rho, out, direction)
           * run_dynamics(dyn_params, Ham, rho_init, adm_init)

.. moduleauthors:: Victor A. Suarez, Alexey V. Akimov


"""

import sys
import time
import types

import numpy as np

try:
    import numba
    from numba import njit, prange
except ImportError as _err:
    raise ImportError("The \"numba\" backend of the HEOM requires the numba package. Install it, e.g. with\n"
                      "    pip install numba            (for Python 3.9 use: pip install numba==0.60.0)\n"
                      "or use dyn_params[\"backend\"] = \"cpp\".") from _err

if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform == "linux" or sys.platform == "linux2":
    from liblibra_core import *

import util.libutil as comn
import libra_py.units as units
from . import save
from .compute import _default_params, setup_bath_expansion


# ======================= Compiled kernels =======================
#
# The state of the hierarchy is a complex128 array rho[n, a, b] of shape (nn_tot, nquant, nquant),
# where n enumerates the ADMs in the order produced by gen_hierarchy. The multi-index of the n-th
# ADM is nvec[n, j], with j = m * nterms + k combining the system-bath coupling term m and the
# term k of the expansion of the bath correlation function. vplus[n, j] (vminus[n, j]) is the
# index of the ADM with the j-th element of the multi-index raised (lowered) by 1, or -1 if that
# ADM is not part of the hierarchy.


# Note: the kernels below index the arrays element by element (rho[n, a, b]) rather than through
# sub-array views (rho[n]); every view in Numba carries a reference-count update of the parent
# array, and these atomic updates would dominate the cost and serialize the threads.

@njit(cache=True)
def _deriv_one(n, rho, out, Ham, Q, gamma, c, cconj, nvec, vplus, vminus, plus_tab, minus_tab,
               trunc_on, trunc_pref, nonzero, S1, S2, T1):
    """
    Computes out[n] = d rho[n] / dt - a direct port of compute_deriv_n in heom.cpp, which follows
    Eq. 15 of JCP 131, 094502 (2009), with the operations done in the same order.

    S1, S2, T1 are (nquant, nquant) scratch matrices
    """

    nq = rho.shape[1]
    nt = gamma.shape[0]

    for a in range(nq):
        for b in range(nq):
            out[n, a, b] = 0.0

    if nonzero[n] != 0:

        # ============= Liouvillian:  -i [H, rho_n] =============
        for a in range(nq):
            for b in range(nq):
                hr = 0.0j
                rh = 0.0j
                for e in range(nq):
                    hr += Ham[a, e] * rho[n, e, b]
                    rh += rho[n, a, e] * Ham[e, b]
                out[n, a, b] = -1.0j * (hr - rh)

        # ============= Friction:  - sum_{m,k} n_mk gamma_k rho_n =============
        pref = 0.0j
        for k in range(nt):
            sum_over_m = 0
            for m in range(nq):
                sum_over_m += nvec[n, m * nt + k]
            pref += float(sum_over_m) * gamma[k]
        for a in range(nq):
            for b in range(nq):
                out[n, a, b] -= pref * rho[n, a, b]

        # ============= Truncation:  - pref * [Q_m, [Q_m, rho_n]] =============
        if trunc_on:
            for m in range(nq):
                # T1 = [Q_m, rho_n]
                for a in range(nq):
                    for b in range(nq):
                        qr = 0.0j
                        rq = 0.0j
                        for e in range(nq):
                            qr += Q[m, a, e] * rho[n, e, b]
                            rq += rho[n, a, e] * Q[m, e, b]
                        T1[a, b] = qr - rq
                # out[n] -= pref * [Q_m, T1]
                for a in range(nq):
                    for b in range(nq):
                        qt = 0.0j
                        tq = 0.0j
                        for e in range(nq):
                            qt += Q[m, a, e] * T1[e, b]
                            tq += T1[a, e] * Q[m, e, b]
                        out[n, a, b] -= trunc_pref * (qt - tq)

    # ===================== rho_n_plus terms =====================
    for m in range(nq):
        for a in range(nq):
            for b in range(nq):
                S1[a, b] = 0.0
        for k in range(nt):
            j = m * nt + k
            nplus = vplus[n, j]
            if nplus != -1:
                if nonzero[nplus] != 0:
                    sf = plus_tab[k, nvec[n, j]]
                    for a in range(nq):
                        for b in range(nq):
                            S1[a, b] += sf * rho[nplus, a, b]
        # out[n] -= i [Q_m, S1]
        for a in range(nq):
            for b in range(nq):
                qs = 0.0j
                sq = 0.0j
                for e in range(nq):
                    qs += Q[m, a, e] * S1[e, b]
                    sq += S1[a, e] * Q[m, e, b]
                out[n, a, b] -= 1.0j * (qs - sq)

    # ===================== rho_n_minus terms =====================
    for m in range(nq):
        for a in range(nq):
            for b in range(nq):
                S1[a, b] = 0.0
                S2[a, b] = 0.0
        for k in range(nt):
            j = m * nt + k
            nminus = vminus[n, j]
            if nminus != -1:
                if nonzero[nminus] != 0:
                    sf = minus_tab[k, nvec[n, j]]
                    # S1 carries C(t), S2 carries C(t)* - see the note on vec_conj in heom.cpp
                    c1 = sf * c[k]
                    c2 = sf * cconj[k]
                    for a in range(nq):
                        for b in range(nq):
                            S1[a, b] += c1 * rho[nminus, a, b]
                            S2[a, b] += c2 * rho[nminus, a, b]
        # out[n] -= i (Q_m S1 - S2 Q_m)
        for a in range(nq):
            for b in range(nq):
                qs = 0.0j
                sq = 0.0j
                for e in range(nq):
                    qs += Q[m, a, e] * S1[e, b]
                    sq += S2[a, e] * Q[m, e, b]
                out[n, a, b] -= 1.0j * (qs - sq)


@njit(cache=True)
def _deriv_one_diag(n, rho, out, Ham, lw, beta, alpha, gamma, nvec, vplus, vminus, plus_tab, minus_tab,
                    nonzero, S1):
    """
    The same as _deriv_one, specialized to diagonal system-bath coupling operators,
    Q_m = diag(qd[m, :]) - e.g. the default projectors Q_m = |m><m|. Then all of the commutators
    reduce to element-wise products, and the coefficients can be folded together in advance
    (see setup_heom):

        lw[a, b]       = trunc_pref * sum_m (qd[m, a] - qd[m, b])^2        (0 without truncation)
        beta[m, a, b]  = -i (qd[m, a] - qd[m, b])
        alpha[j, a, b] = -i (qd[m, a] c_k - conj(c_vec_conj[k]) qd[m, b]),   j = m * nterms + k

    so that

        out[n]_ab = -i [H, rho_n]_ab - (sum_{m,k} n_mk gamma_k + lw_ab) rho_n_ab
                    + sum_m beta[m]_ab sum_k plus_tab[k, n_mk]  rho_{n+e_j}_ab
                    + sum_j minus_tab[k, n_mk] alpha[j]_ab rho_{n-e_j}_ab

    This changes the order of the floating-point operations relative to compute_deriv_n, so the
    results agree with it to rounding (~1e-15 relative), rather than bit for bit.
    """

    nq = rho.shape[1]
    nt = gamma.shape[0]

    if nonzero[n] != 0:

        # ============= Friction:  - sum_{m,k} n_mk gamma_k rho_n =============
        pref = 0.0j
        for k in range(nt):
            sum_over_m = 0
            for m in range(nq):
                sum_over_m += nvec[n, m * nt + k]
            pref += float(sum_over_m) * gamma[k]

        # ============= Liouvillian, friction, and truncation =============
        for a in range(nq):
            for b in range(nq):
                hr = 0.0j
                rh = 0.0j
                for e in range(nq):
                    hr += Ham[a, e] * rho[n, e, b]
                    rh += rho[n, a, e] * Ham[e, b]
                out[n, a, b] = -1.0j * (hr - rh) - (pref + lw[a, b]) * rho[n, a, b]
    else:
        for a in range(nq):
            for b in range(nq):
                out[n, a, b] = 0.0

    for m in range(nq):
        for a in range(nq):
            for b in range(nq):
                S1[a, b] = 0.0
        for k in range(nt):
            j = m * nt + k
            nj = nvec[n, j]

            # ===================== rho_n_plus terms =====================
            nplus = vplus[n, j]
            if nplus != -1:
                if nonzero[nplus] != 0:
                    sf = plus_tab[k, nj]
                    for a in range(nq):
                        for b in range(nq):
                            S1[a, b] += sf * rho[nplus, a, b]

            # ===================== rho_n_minus terms =====================
            nminus = vminus[n, j]
            if nminus != -1:
                if nonzero[nminus] != 0:
                    sf = minus_tab[k, nj]
                    for a in range(nq):
                        for b in range(nq):
                            out[n, a, b] += (sf * alpha[j, a, b]) * rho[nminus, a, b]

        for a in range(nq):
            for b in range(nq):
                out[n, a, b] += beta[m, a, b] * S1[a, b]


@njit(parallel=True, cache=True)
def _derivs_diag2(rho, out, Ham, lw, beta, alpha, gamma, nvec, vplus, vminus, plus_tab, minus_tab,
                  nonzero, active):
    """
    _derivs_diag for two states (nquant = 2), with the 2x2 algebra written out explicitly - the most
    common case, and about twice as fast as the general kernel
    """

    nt = gamma.shape[0]
    h00 = Ham[0, 0]
    h01 = Ham[0, 1]
    h10 = Ham[1, 0]
    h11 = Ham[1, 1]
    w00 = lw[0, 0]
    w01 = lw[0, 1]
    w10 = lw[1, 0]
    w11 = lw[1, 1]

    for i in prange(active.shape[0]):
        n = active[i]
        o00 = 0.0j
        o01 = 0.0j
        o10 = 0.0j
        o11 = 0.0j

        if nonzero[n] != 0:
            r00 = rho[n, 0, 0]
            r01 = rho[n, 0, 1]
            r10 = rho[n, 1, 0]
            r11 = rho[n, 1, 1]

            pref = 0.0j
            for k in range(nt):
                pref += float(nvec[n, k] + nvec[n, nt + k]) * gamma[k]

            o00 = -1.0j * ((h00 * r00 + h01 * r10) - (r00 * h00 + r01 * h10)) - (pref + w00) * r00
            o01 = -1.0j * ((h00 * r01 + h01 * r11) - (r00 * h01 + r01 * h11)) - (pref + w01) * r01
            o10 = -1.0j * ((h10 * r00 + h11 * r10) - (r10 * h00 + r11 * h10)) - (pref + w10) * r10
            o11 = -1.0j * ((h10 * r01 + h11 * r11) - (r10 * h01 + r11 * h11)) - (pref + w11) * r11

        for m in range(2):
            p00 = 0.0j
            p01 = 0.0j
            p10 = 0.0j
            p11 = 0.0j
            for k in range(nt):
                j = m * nt + k
                nj = nvec[n, j]

                nplus = vplus[n, j]
                if nplus != -1:
                    if nonzero[nplus] != 0:
                        sf = plus_tab[k, nj]
                        p00 += sf * rho[nplus, 0, 0]
                        p01 += sf * rho[nplus, 0, 1]
                        p10 += sf * rho[nplus, 1, 0]
                        p11 += sf * rho[nplus, 1, 1]

                nminus = vminus[n, j]
                if nminus != -1:
                    if nonzero[nminus] != 0:
                        sf = minus_tab[k, nj]
                        o00 += (sf * alpha[j, 0, 0]) * rho[nminus, 0, 0]
                        o01 += (sf * alpha[j, 0, 1]) * rho[nminus, 0, 1]
                        o10 += (sf * alpha[j, 1, 0]) * rho[nminus, 1, 0]
                        o11 += (sf * alpha[j, 1, 1]) * rho[nminus, 1, 1]

            o00 += beta[m, 0, 0] * p00
            o01 += beta[m, 0, 1] * p01
            o10 += beta[m, 1, 0] * p10
            o11 += beta[m, 1, 1] * p11

        out[n, 0, 0] = o00
        out[n, 0, 1] = o01
        out[n, 1, 0] = o10
        out[n, 1, 1] = o11


@njit(parallel=True, cache=True)
def _derivs(rho, out, Ham, Q, gamma, c, cconj, nvec, vplus, vminus, plus_tab, minus_tab,
            trunc_on, trunc_pref, nonzero, active, nchunks):
    """
    out[n] = d rho[n] / dt  for all n in `active`. The rows of `out` that are not listed in
    `active` are not touched - the caller is responsible for zeroing them, if needed.
    """

    nact = active.shape[0]
    nq = rho.shape[1]
    chunk = (nact + nchunks - 1) // nchunks

    for ic in prange(nchunks):
        S1 = np.empty((nq, nq), dtype=np.complex128)
        S2 = np.empty((nq, nq), dtype=np.complex128)
        T1 = np.empty((nq, nq), dtype=np.complex128)
        i0 = ic * chunk
        i1 = min(nact, i0 + chunk)
        for i in range(i0, i1):
            _deriv_one(active[i], rho, out, Ham, Q, gamma, c, cconj, nvec, vplus, vminus,
                       plus_tab, minus_tab, trunc_on, trunc_pref, nonzero, S1, S2, T1)


@njit(parallel=True, cache=True)
def _derivs_diag(rho, out, Ham, lw, beta, alpha, gamma, nvec, vplus, vminus, plus_tab, minus_tab,
                 nonzero, active, nchunks):
    """ The same as _derivs, for diagonal coupling operators - see _deriv_one_diag """

    nact = active.shape[0]
    nq = rho.shape[1]
    chunk = (nact + nchunks - 1) // nchunks

    for ic in prange(nchunks):
        S1 = np.empty((nq, nq), dtype=np.complex128)
        i0 = ic * chunk
        i1 = min(nact, i0 + chunk)
        for i in range(i0, i1):
            _deriv_one_diag(active[i], rho, out, Ham, lw, beta, alpha, gamma, nvec, vplus, vminus,
                            plus_tab, minus_tab, nonzero, S1)


@njit(parallel=True, cache=True)
def _rk_stage(y, k, acc, tmp, a, b, first):
    """
    One fused pass over the flattened state:
        acc  = y + b * k   (if first)   or   acc += b * k   (otherwise)
        tmp  = y + a * k
    """

    for i in prange(y.shape[0]):
        kv = k[i]
        if first:
            acc[i] = y[i] + b * kv
        else:
            acc[i] += b * kv
        tmp[i] = y[i] + a * kv


@njit(parallel=True, cache=True)
def _rk_last(k, acc, b):
    """ acc += b * k  over the flattened state """

    for i in prange(k.shape[0]):
        acc[i] += b * k[i]


@njit(parallel=True, cache=True)
def _scale_rows(x, fac, out):
    """ out[n] = x[n] * fac[n] """

    nq = x.shape[1]
    for n in prange(x.shape[0]):
        f = fac[n]
        for a in range(nq):
            for b in range(nq):
                out[n, a, b] = x[n, a, b] * f


@njit(parallel=True, cache=True)
def _flag_above(x, tol, flags):
    """
    flags[n] = 1 if max_{a,b} |x[n, a, b]| > tol, and 0 otherwise - the same test as
    abs( x[n].max_elt() ) > tol for CMATRIX. Returns the number of flagged rows.

    The complex modulus (a hypot call) is only evaluated when the cheap bounds
    max(|Re|, |Im|) <= |z| <= sqrt(2) * max(|Re|, |Im|) don't already decide the comparison
    """

    nq = x.shape[1]
    count = 0
    for n in prange(x.shape[0]):
        f = 0
        for a in range(nq):
            for b in range(nq):
                z = x[n, a, b]
                mm = max(abs(z.real), abs(z.imag))
                if mm > tol:
                    f = 1
                elif mm * 1.5 > tol:
                    if abs(z) > tol:
                        f = 1
        flags[n] = f
        count += f
    return count


@njit(parallel=True, cache=True)
def _zero_rows(x, mask):
    """ x[n] = 0 for all n with mask[n] == 0 """

    nq = x.shape[1]
    for n in prange(x.shape[0]):
        if mask[n] == 0:
            for a in range(nq):
                for b in range(nq):
                    x[n, a, b] = 0.0


@njit(parallel=True, cache=True)
def _scaling_factors(nvec, fact_tab, pow_tab, nquant, nterms):
    """
    The factors of the scaled HEOM (JCP 130, 084105, 2009), computed with the same products as in
    scale_rho of heom.cpp:  sf[n] = prod_{m,k}  n_mk! * |c_k|^n_mk
    """

    nn = nvec.shape[0]
    sf = np.empty(nn, dtype=np.float64)
    for n in prange(nn):
        s = 1.0
        for m in range(nquant):
            for k in range(nterms):
                nn_mk = nvec[n, m * nterms + k]
                s *= fact_tab[nn_mk]
                s *= pow_tab[k, nn_mk]
        sf[n] = s
    return sf


# ======================= Setup =======================

_topology_cache = {}


def _intList2_to_np(x, d):
    return np.fromiter((i for v in x for i in v), dtype=np.int32, count=len(x) * d).reshape(len(x), d)


def build_topology(d, LL, verbosity=-1):
    """
    Generates the hierarchy with the C++ gen_hierarchy and converts it to numpy arrays.
    The results are cached, so that repeated calls (e.g. a thermalization run followed by
    a dynamics run) don't regenerate the hierarchy.

    Args:
        d ( int ): the length of the multi-index, nquant * nterms
        LL ( int ): the maximal tier of the hierarchy
        verbosity ( int ): passed to gen_hierarchy

    Returns:
        tuple: (nvec, vplus, vminus) - int32 arrays of shape (nn_tot, d)

    """

    key = (d, LL)
    if key not in _topology_cache:
        all_vectors, vec_plus, vec_minus = intList2(), intList2(), intList2()
        gen_hierarchy(d, LL, verbosity, all_vectors, vec_plus, vec_minus)
        nvec = _intList2_to_np(all_vectors, d)
        del all_vectors
        vplus = _intList2_to_np(vec_plus, d)
        del vec_plus
        vminus = _intList2_to_np(vec_minus, d)
        del vec_minus
        _topology_cache.clear()  # keep only one hierarchy around
        _topology_cache[key] = (nvec, vplus, vminus)

    return _topology_cache[key]


def compute_truncation_prefactor(params, gamma_matsubara, c_matsubara, gamma_matsubara_ext, c_matsubara_ext):
    """
    The prefactor of the [Q, [Q, rho_n]] truncation term - a port of the corresponding part
    of compute_heom_derivatives in heom.cpp

    Returns:
        (bool, complex): whether the truncation term is used, and its prefactor

    """

    scheme = params["truncation_scheme"]
    kB = units.boltzmann / units.hartree
    last = len(gamma_matsubara) - 1
    pref = 0.0j

    if scheme == 1 or scheme == 2:
        # Ihizaki-Tanimura scheme for truncation, JPSJ 74 3131, 2005
        pref = params["eta"] * kB * params["temperature"] / complex(gamma_matsubara[0])
        if scheme == 1:
            # according to Schulten, with real part of Matsubara terms
            pref -= complex(compute_matsubara_sum(gamma_matsubara, c_matsubara, last)).real
        if scheme == 2:
            # according to Schulten, but with full Matsubara terms
            pref -= complex(compute_matsubara_sum(gamma_matsubara, c_matsubara, last))

    if scheme == 3 or scheme == 4:
        sum_KK = complex(compute_matsubara_sum(gamma_matsubara, c_matsubara, last))
        sum_KK_ext = complex(compute_matsubara_sum(gamma_matsubara_ext, c_matsubara_ext, len(gamma_matsubara_ext) - 1))
        if scheme == 3:
            pref = sum_KK_ext - sum_KK
        if scheme == 4:
            pref = complex((sum_KK_ext - sum_KK).real, 0.0)

    return (scheme in (1, 2, 3, 4)), complex(pref)


def _cmatrix_to_np(X):
    if isinstance(X, np.ndarray):
        return np.array(X, dtype=np.complex128)
    return np.array([[X.get(i, j) for j in range(X.num_of_cols)] for i in range(X.num_of_rows)],
                    dtype=np.complex128)


def setup_heom(params, Ham):
    """
    Builds everything that is needed to evaluate the HEOM derivatives with the compiled kernels

    Args:
        params ( dict ): the parameters of `run_dynamics`, already completed with the defaults
        Ham ( CMATRIX(nquant, nquant) or ndarray ): the system's Hamiltonian

    Returns:
        SimpleNamespace: with the topology of the hierarchy (nvec, vplus, vminus), the arrays of
            coefficients of the equations, and the scaling factors of the scaled HEOM

    """

    h = types.SimpleNamespace()

    h.Ham = _cmatrix_to_np(Ham)
    nquant = h.Ham.shape[0]
    h.nquant = nquant

    KK = params["KK"] if params["KK"] >= 0 else 0
    LL = params["LL"]
    h.LL = LL

    gm, cm, gme, cme, vec_conj = setup_bath_expansion(params, KK)
    h.gamma_matsubara, h.c_matsubara = gm, cm
    nterms = len(gm)
    h.nterms = nterms

    h.gamma = np.array([complex(gm[k]) for k in range(nterms)], dtype=np.complex128)
    h.c = np.array([complex(cm[k]) for k in range(nterms)], dtype=np.complex128)
    h.vec_conj = np.array(vec_conj, dtype=np.int64)
    h.cconj = np.conj(h.c[h.vec_conj])

    h.trunc_on, h.trunc_pref = compute_truncation_prefactor(params, gm, cm, gme, cme)

    # Only the first nquant coupling operators enter the equations - see compute_deriv_n
    epc = params["el_phon_couplings"]
    h.Q = np.array([_cmatrix_to_np(epc[m]) for m in range(nquant)], dtype=np.complex128)

    # Diagonal coupling operators (such as the default |m><m| projectors) allow cheaper kernels,
    # with the coefficients folded together in advance - see _deriv_one_diag
    h.diagonal_Q = all(np.count_nonzero(h.Q[m] - np.diag(np.diag(h.Q[m]))) == 0 for m in range(nquant))
    qd = np.array([np.diag(h.Q[m]) for m in range(nquant)], dtype=np.complex128)
    h.lw = np.zeros((nquant, nquant), dtype=np.complex128)
    if h.trunc_on:
        for m in range(nquant):
            f = qd[m][:, None] - qd[m][None, :]
            h.lw += f * f
        h.lw *= h.trunc_pref
    h.beta = np.array([-1.0j * (qd[m][:, None] - qd[m][None, :]) for m in range(nquant)], dtype=np.complex128)
    h.alpha = np.array([-1.0j * (qd[m][:, None] * h.c[k] - h.cconj[k] * qd[m][None, :])
                        for m in range(nquant) for k in range(nterms)], dtype=np.complex128)

    # ============== Topology ==============
    d = nquant * nterms
    h.nvec, h.vplus, h.vminus = build_topology(d, LL, params["verbosity"])
    h.nn_tot = h.nvec.shape[0]

    # ============== Coefficients of the rho_n+ and rho_n- terms ==============
    # tab[k, n_mk] - the factor that multiplies rho[n+] (rho[n-]) in the equation for rho[n]
    absc = np.abs(h.c)
    nmax = LL + 1
    nn_mk = np.arange(nmax + 1, dtype=np.float64)
    h.plus_tab = np.ones((nterms, nmax + 1), dtype=np.float64)
    h.minus_tab = np.zeros((nterms, nmax + 1), dtype=np.float64)
    for k in range(nterms):
        if params["do_scale"] == 1:
            # Scaled HEOM approach from JCP 130, 084105 (2009)
            h.plus_tab[k, :] = np.sqrt((nn_mk + 1.0) * absc[k])
            if absc[k] > 1e-50:
                h.minus_tab[k, :] = np.sqrt(nn_mk / absc[k])
        else:
            h.minus_tab[k, :] = nn_mk

    # ============== Scaling factors ==============
    h.do_scale = params["do_scale"]
    if h.do_scale == 1:
        fact_tab = np.array([FACTORIAL(i) for i in range(LL + 1)], dtype=np.float64)
        pow_tab = np.array([[FAST_POW(float(absc[k]), i) for i in range(LL + 1)] for k in range(nterms)],
                           dtype=np.float64)
        sf = _scaling_factors(h.nvec, fact_tab, pow_tab, nquant, nterms)
        h.fwd_fac = np.sqrt(1.0 / sf)   # raw -> scaled, as in scale_rho
        h.inv_fac = np.sqrt(sf)         # scaled -> raw, as in inv_scale_rho
    else:
        h.fwd_fac = None
        h.inv_fac = None

    h.all_indices = np.arange(h.nn_tot, dtype=np.int64)
    h.nchunks = max(1, 8 * numba.get_num_threads())

    return h


def compute_derivatives(heom, rho, out, nonzero, active=None):
    """
    out = d rho / dt  -  the equivalent of compute_heom_derivatives, for the scaled or the unscaled
    HEOM (as selected by params["do_scale"] passed to setup_heom)

    Args:
        heom ( SimpleNamespace ): as returned by setup_heom
        rho ( ndarray(nn_tot, nquant, nquant) ): the hierarchy
        out ( ndarray(nn_tot, nquant, nquant) ): the result - the derivatives of all ADMs listed in
            `active`, and zeros for all the other ADMs
        nonzero ( ndarray(nn_tot) of uint8 ): 1 for the ADMs that enter the equations, 0 for those
            that are discarded
        active ( ndarray of int64 or None ): the indices of the ADMs whose derivatives are computed;
            None means all of them

    """

    if active is None:
        active = heom.all_indices
    elif active.shape[0] < heom.nn_tot:
        out[:] = 0.0

    if heom.diagonal_Q and heom.nquant == 2:
        _derivs_diag2(rho, out, heom.Ham, heom.lw, heom.beta, heom.alpha, heom.gamma, heom.nvec,
                      heom.vplus, heom.vminus, heom.plus_tab, heom.minus_tab, nonzero, active)
    elif heom.diagonal_Q:
        _derivs_diag(rho, out, heom.Ham, heom.lw, heom.beta, heom.alpha, heom.gamma, heom.nvec,
                     heom.vplus, heom.vminus, heom.plus_tab, heom.minus_tab, nonzero, active, heom.nchunks)
    else:
        _derivs(rho, out, heom.Ham, heom.Q, heom.gamma, heom.c, heom.cconj, heom.nvec, heom.vplus, heom.vminus,
                heom.plus_tab, heom.minus_tab, heom.trunc_on, heom.trunc_pref, nonzero, active, heom.nchunks)


def scale_adms(heom, rho, out, direction):
    """
    direction:
     1 :  raw -> scaled   (as scale_rho in heom.cpp)
    -1 :  scaled -> raw   (as inv_scale_rho in heom.cpp)

    Without the scaling (params["do_scale"] = 0) this is just a copy
    """

    if heom.do_scale != 1:
        out[:] = rho
    elif direction == 1:
        _scale_rows(rho, heom.fwd_fac, out)
    else:
        _scale_rows(rho, heom.inv_fac, out)


def _to_adm_array(adm_init, nn_tot, nquant):
    if isinstance(adm_init, np.ndarray):
        adm = np.asarray(adm_init, dtype=np.complex128)
    else:
        adm = np.array([_cmatrix_to_np(adm_init[n]) for n in range(len(adm_init))], dtype=np.complex128)
    assert adm.shape == (nn_tot - 1, nquant, nquant), \
        f"adm_init must hold nn_tot-1 = {nn_tot - 1} matrices of size {nquant}x{nquant}, got the shape {adm.shape}."
    return adm


def _print_matrices(x):
    for imat in range(x.shape[0]):
        print(F"== imat = {imat} ==")
        for row in range(x.shape[1]):
            print("  ".join(F"{x[imat, row, col]}" for col in range(x.shape[2])))


def run_dynamics(dyn_params, Ham, rho_init, adm_init=None):
    """
    Integrates the HEOM for a given system's Hamiltonian, initial conditions, and bath parameters.

    The parameters, the output files, and the returned value are the same as for
    `libra_py.dynamics.heom.compute.run_dynamics` - see its documentation. In addition:

        * `rho_init` can be a CMATRIX(nstates, nstates) or an ndarray(nstates, nstates)
        * `adm_init` can be a CMATRIXList or an ndarray(nn_tot-1, nstates, nstates)
        * dyn_params["num_threads"] sets the number of Numba threads

    """

    params = dict(dyn_params)
    critical_params = []
    default_params = _default_params(Ham)
    comn.check_input(params, default_params, critical_params)

    nthreads = max(1, min(int(params["num_threads"]), numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(nthreads)

    nsteps = params["nsteps"]
    print_freq = max(1, int(params["progress_frequency"] * nsteps))
    dt = params["dt"]

    # ============= System, bath, and hierarchy ======================
    heom = setup_heom(params, Ham)
    nquant, nterms, nn_tot = heom.nquant, heom.nterms, heom.nn_tot

    if params["verbosity"] >= 1:
        for k in range(nterms):
            print(F" k = {k} gamma_matsubara[{k}] = {heom.gamma[k]}  c_matsubara[{k}] = {heom.c[k]}")

    if params["verbosity"] >= 2 and heom.do_scale == 1:
        print("Scaling factors")
        for n in range(nn_tot):
            for m in range(nquant):
                for k in range(nterms):
                    n_mk = heom.nvec[n, m * nterms + k]
                    scl = 1.0 / np.sqrt(FACTORIAL(int(n_mk)) * FAST_POW(abs(heom.c[k]), int(n_mk)))
                    print(F" n={n} m={m} k={k}  scaling_factor={scl}")

    # ============= Initialization ============
    # Four buffers for the RK4: the state y, the stage input tmp, the stage derivative k, and the
    # accumulated result acc. Their flat views are used by the element-wise kernels.
    y = np.zeros((nn_tot, nquant, nquant), dtype=np.complex128)
    tmp = np.zeros_like(y)
    k = np.zeros_like(y)
    acc = np.zeros_like(y)

    # raw initial conditions go into tmp, then get scaled into y
    tmp[0] = _cmatrix_to_np(rho_init)
    if adm_init is not None:
        tmp[1:] = _to_adm_array(adm_init, nn_tot, nquant)
    scale_adms(heom, tmp, y, 1)

    adm_tol = params["adm_tolerance"]
    deriv_tol = params["adm_deriv_tolerance"]
    do_zeroing = params["do_zeroing"]
    deriv_flags = np.empty(nn_tot, dtype=np.uint8)

    def update_filters(y, nonzero):
        """
        The same as update_filters in compute.py: the derivatives of all ADMs are computed (into k)
        with the current `nonzero`, then the ADMs with |dADM/dt| > adm_deriv_tolerance define the
        active equations, and the ADMs with |ADM| > adm_tolerance define the new `nonzero`.
        Returns the new (nonzero, active); active is None when all of the equations are active, and
        otherwise deriv_flags marks the active equations
        """

        compute_derivatives(heom, y, k, nonzero, None)

        nactive = _flag_above(k, deriv_tol, deriv_flags)
        active = None if nactive == nn_tot else np.flatnonzero(deriv_flags).astype(np.int64)

        new_nonzero = np.empty(nn_tot, dtype=np.uint8)
        nnonzero = _flag_above(y, adm_tol, new_nonzero)
        if do_zeroing == 1 and nnonzero < nn_tot:
            _zero_rows(y, new_nonzero)

        return new_nonzero, active

    # ========== Filter scaled ADMs ======================
    nonzero = np.ones(nn_tot, dtype=np.uint8)
    nonzero, active = update_filters(y, nonzero)

    if params["verbosity"] >= 2:
        print("nonzero = ", nonzero.tolist())
        print("adm_list = ", (heom.all_indices if active is None else active).tolist())

        if params["verbosity"] >= 4:
            print("ADMs")
            scale_adms(heom, y, tmp, -1)
            _print_matrices(tmp)
            print("Scaled ADMs")
            _print_matrices(y)

    # Initialize savers
    _savers = save.init_heom_savers(params, nquant, nn_tot)

    def get_aux():
        # scaled -> raw for the whole hierarchy; tmp is free at this point of the step
        scale_adms(heom, y, tmp, -1)
        return tmp[1:]

    # ============== Propagation =============
    start = time.time()
    for step in range(nsteps):

        # ================ Saving and printout ===================
        # The 0-th ADM is the same in the scaled and in the raw representation
        save.save_heom_data_np(_savers, step, params, y[0].copy(), get_aux)

        if step % print_freq == 0:
            print(F" step= {step}")

            if params["verbosity"] >= 3:
                print("nonzero = ", nonzero.tolist())
                print("adm_list = ", (heom.all_indices if active is None else active).tolist())

            if params["verbosity"] >= 4:
                print("ADMs")
                _print_matrices(get_aux())
                print("Scaled ADMs")
                _print_matrices(y)

        # ============== Update the list of active equations = Filtering  ============
        k1_ready = False
        if step % params["filter_after_steps"] == 0:
            old_nonzero = nonzero
            nonzero, active = update_filters(y, nonzero)

            # The filtering has just computed (into k) the derivatives of all ADMs using the old
            # `nonzero`. If `nonzero` did not change, these are exactly the derivatives the first RK4
            # stage would compute, except for the rows outside the new list of active equations,
            # which the first RK4 stage would set to zero. Any ADM zeroed by the filter has
            # nonzero = 0 in both lists, so it doesn't enter either calculation.
            if np.array_equal(nonzero, old_nonzero):
                if active is not None:
                    _zero_rows(k, deriv_flags)
                k1_ready = True

        # ================= Propagation for one timestep ==================================
        # y(t+dt) = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        yf, kf, accf, tmpf = y.reshape(-1), k.reshape(-1), acc.reshape(-1), tmp.reshape(-1)

        if not k1_ready:
            compute_derivatives(heom, y, k, nonzero, active)
        _rk_stage(yf, kf, accf, tmpf, 0.5 * dt, dt / 6.0, True)

        compute_derivatives(heom, tmp, k, nonzero, active)
        _rk_stage(yf, kf, accf, tmpf, 0.5 * dt, dt / 3.0, False)

        compute_derivatives(heom, tmp, k, nonzero, active)
        _rk_stage(yf, kf, accf, tmpf, dt, dt / 3.0, False)

        compute_derivatives(heom, tmp, k, nonzero, active)
        _rk_last(kf, accf, dt / 6.0)

        y, acc = acc, y

    end = time.time()
    print(F"Calculations took {end - start} seconds")

    # For the mem_saver - store all the results into HDF5 format only at the end of the simulation
    if _savers["mem_saver"] is not None:
        prefix = params["prefix"]
        _savers["mem_saver"].save_data(F"{prefix}/mem_data.hdf", params["properties_to_save"], "w")
        return _savers["mem_saver"]

