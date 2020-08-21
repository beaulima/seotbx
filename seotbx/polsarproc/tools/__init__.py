import logging
import numpy as np

logger = logging.getLogger("seotbx.polsarproc.sim")


def polsar_one_look_simulation(M_in, n_samples: int):
    """
    Produce n_samples one look simulations from the input covariance matrix (M_in)
    from:
        Title: Generation of Pol-SAR and POL-in-SAR Data for Homogeneous Distributed Targets Simulation
        Authors: Pipia, L. & Fàbregas, X.
        Journal: Proceedings of the 2nd International Workshop POLINSAR 2005 (ESA SP-586). 17 - 21 January 2005,
        ESRIN, Frascati, Italy. Editor: H. Lacoste. ISBN: 92-9092-897-2., p.35
    Args:
        M_in (complex array 3x3): covariance or coherence matrix.
        n_samples (int): Number of output samples.

    Returns:
        complex array n_samplesx3x3: one look simulation

    """
    mean = 0.0
    var = 0.5
    scale = np.sqrt(var)
    L = np.linalg.cholesky(M_in)
    # simulate a complex random vector k that is complex normal distributed with zero mean and 0.5 variance
    # (Shh, Shv, Svv)
    k = np.random.normal(loc=mean, scale=scale, size=(n_samples, 3, 1)) + 1j * np.random.normal(loc=mean, scale=scale,
                                                                                                size=(n_samples, 3, 1))
    # one look (Shh, Shv, Svv)
    kp = np.expand_dims(np.dot(L, k)[:, :, 0].transpose(), axis=1)
    kpt = kp.conjugate().transpose().transpose(2, 0, 1)
    # return simulate coherence/covariance matrix
    return kp[:] * kpt[:]


def polsar_n_looks_simulation(M_in, n_looks: int, n_samples: int):
    """
       Produce n_samples one look simulations from the input covariance matrix (M_in)
       from:
           Title: Generation of Pol-SAR and POL-in-SAR Data for Homogeneous Distributed Targets Simulation
           Authors: Pipia, L. & Fàbregas, X.
           Journal: Proceedings of the 2nd International Workshop POLINSAR 2005 (ESA SP-586). 17 - 21 January 2005,
           ESRIN, Frascati, Italy. Editor: H. Lacoste. ISBN: 92-9092-897-2., p.35
       Args:
           M_in (complex array 3x3): covariance or coherence matrix.
           n_looks (int): Number of looks.
           n_samples (int): Number of output samples.

       Returns:
           complex array n_samplesx3x3: n looks simulation

       """
    # prepare output buffer
    M_out = np.zeros((n_samples, 3, 3)) + 1j * np.zeros((n_samples, 3, 3))
    for i_look in range(n_looks):
        # compute one look simulation
        M_out_1look = polsar_one_look_simulation(M_in, n_samples)
        if n_looks == 1:
            return M_out_1look
        # compute multi looks simulation
        M_out += (M_out_1look / n_looks)
    return M_out

def span_normalize_M3(M_in):
    """
    Normalize the matrix by the span
    """
    assert M_in.shape[0] == 3 and M_in.shape[1] == 3
    span = M_in[0,0].real + M_in[1,1].real +  + M_in[2,2].real
    return M_in / span


