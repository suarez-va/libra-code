"""
Tests of the numba backend of the HEOM (libra_py.dynamics.heom.compute_numba) against the
C++ implementation (compute_heom_derivatives, scale_rho, inv_scale_rho, and the "cpp" backend
of compute.run_dynamics).

The tests import libra_py from the build tree (as set by PYTHONPATH), so run
`make copy-libra-py` in the build directory after changing the Python sources.

Run with pytest, or directly:  python test_heom_numba.py
"""

import os
import itertools
import tempfile
import importlib.util

import numpy as np
import h5py

from liblibra_core import *
import util.libutil as comn
import libra_py.dynamics.heom.compute as compute

HAVE_NUMBA = importlib.util.find_spec("numba") is not None

try:
    import pytest
    requires_numba = pytest.mark.skipif(not HAVE_NUMBA, reason="numba is not installed")
except ImportError:
    def requires_numba(f):
        return f

if HAVE_NUMBA:
    import libra_py.dynamics.heom.compute_numba as cn


T_BETA1 = 315774.6670514614  # temperature at which beta = 1 a.u.


# ===================== helpers =====================

def _np_to_cmatrix(x):
    X = CMATRIX(x.shape[0], x.shape[1])
    for a in range(x.shape[0]):
        for b in range(x.shape[1]):
            X.set(a, b, complex(x[a, b]))
    return X


def _stack_to_cmatrix(rho):
    nn, nq, _ = rho.shape
    R = CMATRIX(nn * nq, nq)
    for n in range(nn):
        for a in range(nq):
            for b in range(nq):
                R.set(n * nq + a, b, complex(rho[n, a, b]))
    return R


def _cmatrix_to_stack(R, nq):
    nn = R.num_of_rows // nq
    return np.array([[[R.get(n * nq + a, b) for b in range(nq)] for a in range(nq)] for n in range(nn)])


def _random_hermitian(rng, nq, scale=1.0):
    x = rng.normal(size=(nq, nq)) + 1j * rng.normal(size=(nq, nq))
    return scale * 0.5 * (x + x.conj().T)


def _bath_params(bath):
    """ (spectral_density, KK, gamma, omega) for the bath types tested """
    return {"debye": (0, 2, 2.0, 0.0),
            "brownian_underdamped": (1, 1, 2.0, 2.0),
            "brownian_overdamped": (1, 1, 6.0, 1.0),
            "brownian_overdamped_KKm1": (1, -1, 6.0, 1.0)}[bath]


def _setup_case(bath, nq, diagonal, do_scale, trunc, LL, seed):
    rng = np.random.default_rng(seed)
    sd, KK, gamma, omega = _bath_params(bath)

    Ham = _np_to_cmatrix(_random_hermitian(rng, nq))
    epc = CMATRIXList()
    for m in range(nq):
        if diagonal:
            q = np.zeros((nq, nq), dtype=complex)
            q[m, m] = 1.0
        else:
            q = _random_hermitian(rng, nq, 0.5)
        epc.append(_np_to_cmatrix(q))

    params = {"KK": KK, "LL": LL, "spectral_density": sd, "gamma": gamma, "omega": omega, "eta": 12.0,
              "temperature": T_BETA1, "el_phon_couplings": epc, "truncation_scheme": trunc,
              "do_scale": do_scale, "verbosity": -1}
    comn.check_input(params, compute._default_params(Ham), [])
    return rng, Ham, params


def _cpp_derivatives(params, Ham, rho, nonzero, adm_list):
    """ d rho / dt from the C++ compute_heom_derivatives, with the inputs prepared as in compute.run_dynamics """

    nq = rho.shape[1]
    KK = params["KK"] if params["KK"] >= 0 else 0
    gm, cm, gme, cme, vec_conj = compute.setup_bath_expansion(params, KK)
    av, vp, vm = intList2(), intList2(), intList2()
    gen_hierarchy(nq * len(gm), params["LL"], -1, av, vp, vm)

    p = dict(params)
    p.update({"Ham": Ham, "gamma_matsubara": gm, "c_matsubara": cm, "gamma_matsubara_ext": gme,
              "c_matsubara_ext": cme, "vec_conj": Py2Cpp_int(vec_conj), "nvec": av, "nvec_plus": vp,
              "nvec_minus": vm, "nonzero": Py2Cpp_int([int(x) for x in nonzero]),
              "adm_list": Py2Cpp_int([int(x) for x in adm_list]), "num_threads": 1})
    D = compute_heom_derivatives(_stack_to_cmatrix(rho), p)
    return _cmatrix_to_stack(D, nq)


# ===================== tests =====================

@requires_numba
def test_topology_matches_gen_hierarchy():
    for d, LL in [(2, 5), (4, 4), (6, 3), (9, 2)]:
        av, vp, vm = intList2(), intList2(), intList2()
        gen_hierarchy(d, LL, -1, av, vp, vm)
        nvec, vplus, vminus = cn.build_topology(d, LL)
        assert nvec.shape == (len(av), d)
        for n in range(len(av)):
            assert list(nvec[n]) == list(av[n])
            assert list(vplus[n]) == list(vp[n])
            assert list(vminus[n]) == list(vm[n])


@requires_numba
def test_derivatives_match_cpp():
    """
    All combinations of bath, scaling, truncation scheme, and coupling operators - the latter select
    the kernel: diagonal with 2 states (unrolled), diagonal with 3 states, and non-diagonal.
    Includes discarded ADMs (nonzero = 0) and a partial list of active equations
    """

    baths = ["debye", "brownian_underdamped", "brownian_overdamped", "brownian_overdamped_KKm1"]
    couplings = [(2, True), (3, True), (2, False), (3, False)]

    for icase, (bath, (nq, diagonal), do_scale, trunc) in enumerate(
            itertools.product(baths, couplings, [0, 1], [0, 1, 2, 3, 4])):

        rng, Ham, params = _setup_case(bath, nq, diagonal, do_scale, trunc, LL=3, seed=icase)
        heom = cn.setup_heom(params, Ham)
        assert heom.diagonal_Q == diagonal
        nn = heom.nn_tot

        rho = rng.normal(size=(nn, nq, nq)) + 1j * rng.normal(size=(nn, nq, nq))

        for partial in (False, True):
            if partial:
                nonzero = (rng.random(nn) < 0.8).astype(np.uint8)
                active = np.sort(rng.choice(nn, size=nn // 2, replace=False)).astype(np.int64)
            else:
                nonzero = np.ones(nn, dtype=np.uint8)
                active = np.arange(nn, dtype=np.int64)

            ref = _cpp_derivatives(params, Ham, rho, nonzero, active)
            out = np.full_like(rho, np.nan)
            cn.compute_derivatives(heom, rho, out, nonzero, None if not partial else active)

            err = np.max(np.abs(out - ref)) / np.max(np.abs(ref))
            assert err < 1e-12, (bath, nq, diagonal, do_scale, trunc, partial, err)


@requires_numba
def test_scaling_matches_cpp():
    for bath in ["debye", "brownian_underdamped"]:
        rng, Ham, params = _setup_case(bath, 2, True, 1, 4, LL=6, seed=7)
        heom = cn.setup_heom(params, Ham)
        nn = heom.nn_tot
        rho = rng.normal(size=(nn, 2, 2)) + 1j * rng.normal(size=(nn, 2, 2))

        KK = params["KK"] if params["KK"] >= 0 else 0
        gm, cm, _, _, _ = compute.setup_bath_expansion(params, KK)
        av, vp, vm = intList2(), intList2(), intList2()
        gen_hierarchy(2 * len(gm), params["LL"], -1, av, vp, vm)
        prms = {"nvec": av, "c_matsubara": cm}

        x = CMATRIXList()
        x_scaled = CMATRIXList()
        for n in range(nn):
            x.append(_np_to_cmatrix(rho[n]))
            x_scaled.append(CMATRIX(2, 2))

        # raw -> scaled
        scale_rho(x, x_scaled, prms)
        ref = np.array([[[x_scaled[n].get(a, b) for b in range(2)] for a in range(2)] for n in range(nn)])
        out = np.empty_like(rho)
        cn.scale_adms(heom, rho, out, 1)
        assert np.max(np.abs(out - ref) / np.abs(ref)) < 1e-14

        # scaled -> raw
        for n in range(nn):
            x_scaled[n] = _np_to_cmatrix(rho[n])
        inv_scale_rho(x, x_scaled, prms)
        ref = np.array([[[x[n].get(a, b) for b in range(2)] for a in range(2)] for n in range(nn)])
        cn.scale_adms(heom, rho, out, -1)
        assert np.max(np.abs(out - ref) / np.abs(ref)) < 1e-14


def _run_two_legs(backend, workdir, tol, filter_after_steps, do_zeroing):
    """ A thermalization leg from a bare density matrix, then a dynamics leg from its full hierarchy """

    Ham0 = _np_to_cmatrix(np.diag([0.5, -0.5]).astype(complex))
    Ham = _np_to_cmatrix(np.array([[0.5, 1.0], [1.0, -0.5]], dtype=complex))

    params = {"KK": 1, "LL": 4, "spectral_density": 1, "gamma": 2.0, "eta": 12.0, "omega": 2.0,
              "temperature": T_BETA1, "el_phon_couplings": initialize_el_phonon_couplings(2),
              "dt": 0.02, "nsteps": 150, "nprint": 149, "verbosity": -1, "progress_frequency": 1.0,
              "truncation_scheme": 4, "do_scale": 1, "adm_tolerance": tol[0], "adm_deriv_tolerance": tol[1],
              "filter_after_steps": filter_after_steps, "do_zeroing": do_zeroing, "num_threads": 1,
              "backend": backend, "prefix": os.path.join(workdir, f"{backend}_therm"),
              "hdf5_output_level": 0, "txt_output_level": 0, "mem_output_level": 4,
              "properties_to_save": ["timestep", "time", "denmat", "auxdenmat"],
              "use_compression": 0, "compression_level": [0, 0, 0]}

    rho0 = CMATRIX(2, 2)
    rho0.set(0, 0, 1.0 + 0.0j)
    compute.run_dynamics(params, Ham0, rho0)

    with h5py.File(os.path.join(params["prefix"], "mem_data.hdf"), "r") as f:
        rho = np.array(f["denmat/data"][-1])
        adm = np.array(f["auxdenmat/data"][-1])
        therm_aux = np.array(f["auxdenmat/data"][:])

    params.update({"dt": 0.01, "nsteps": 100, "nprint": 1, "prefix": os.path.join(workdir, f"{backend}_dyn")})

    if backend == "numba":
        # the numba backend also takes numpy arrays
        rho_init, adm_init = rho, adm
    else:
        rho_init = _np_to_cmatrix(rho)
        adm_init = CMATRIXList()
        for n in range(adm.shape[0]):
            adm_init.append(_np_to_cmatrix(adm[n]))

    compute.run_dynamics(params, Ham, rho_init, adm_init)

    with h5py.File(os.path.join(params["prefix"], "mem_data.hdf"), "r") as f:
        res = {key: np.array(f[F"{key}/data"][:]) for key in ("timestep", "time", "denmat", "auxdenmat")}
    res["therm_aux"] = therm_aux
    return res


@requires_numba
def test_run_dynamics_matches_cpp():
    cases = [((0.0, 0.0), 1, 0),       # no actual filtering - uses the reuse of the filter derivatives
             ((1e-6, 1e-12), 5, 1)]    # real filtering, with zeroing of the discarded ADMs

    for tol, filter_after_steps, do_zeroing in cases:
        with tempfile.TemporaryDirectory() as workdir:
            ref = _run_two_legs("cpp", workdir, tol, filter_after_steps, do_zeroing)
            res = _run_two_legs("numba", workdir, tol, filter_after_steps, do_zeroing)

        assert np.array_equal(res["timestep"], ref["timestep"])
        assert np.array_equal(res["time"], ref["time"])
        for key in ("denmat", "auxdenmat", "therm_aux"):
            err = np.max(np.abs(res[key] - ref[key])) / np.max(np.abs(ref[key]))
            assert err < 1e-12, (tol, key, err)


if __name__ == "__main__":
    for test in [test_topology_matches_gen_hierarchy, test_derivatives_match_cpp,
                 test_scaling_matches_cpp, test_run_dynamics_matches_cpp]:
        test()
        print(F"{test.__name__}: passed")
