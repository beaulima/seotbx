import logging
logger = logging.getLogger("pyatcortbx.polsarproc.decomposition")
import numpy as np
import seotbx
import seotbx.polsarproc.definitions as defs


def t3_haalpha_decomposition(M_in, full_computation: bool = True):
    """
    :param M_in: 3x3x...
    :param full_computation:
    :return:
    """
    assert M_in.shape[0] == 3 and M_in.shape[1] == 3
    sz = np.prod(M_in.shape)
    sz = sz // 9
    original_shape = M_in.shape
    M_in = M_in.reshape(3, 3, sz)
    if full_computation:
        nout = defs.len_keys(defs.HAAlpha_IDX)
    else:
        nout = 3
    finale_shape = [nout]
    if len(original_shape) > 2:
        for d in original_shape[2:len(original_shape)]:
            finale_shape.append(d)
    M_out = np.zeros((nout, sz))
    lambdak, V = np.linalg.eigh(M_in.transpose(2, 0, 1), UPLO='L')
    # Descending order
    lambdak = lambdak[:, [2, 1, 0]].transpose(1, 0)
    V = V[:, :, [2, 1, 0]].transpose(1, 2, 0)
    V0 = V[0, :]
    V1 = V[1, :]
    V2 = V[2, :]
    lambdak[lambdak < 0] = 0.0
    pk = lambdak / (np.sum(lambdak, axis=0) + seotbx.utils.const.EPS)
    pk[pk > 1.0] = 1.0
    pk[pk < 0.0] = 0.0
    alphak = np.arccos(np.absolute(V0)) * seotbx.utils.const.RAD2DEG
    if full_computation:
        betak = np.arctan2(np.absolute(V2), seotbx.utils.const.EPS + np.absolute(V1)) * seotbx.utils.const.RAD2DEG
        phik = np.arctan2(V0.imag, seotbx.utils.const.EPS + V0.real)
        deltak = np.arctan2(V1.imag, seotbx.utils.const.EPS + V1.real) - phik
        deltak = np.arctan2(np.sin(deltak), np.cos(deltak) + seotbx.utils.const.EPS) * seotbx.utils.const.RAD2DEG
        gammak = np.arctan2(V2.imag, seotbx.utils.const.EPS + V2.real) - phik
        gammak = np.arctan2(np.sin(gammak), np.cos(gammak) + seotbx.utils.const.EPS) * seotbx.utils.const.RAD2DEG
    Entropy = -np.sum(pk * np.log(pk + seotbx.utils.const.EPS), axis=0) / seotbx.utils.const.LOG3
    Entropy[Entropy < 0] = 0.0
    Anisotropy = (pk[1] - pk[2]) / (pk[1] + pk[2] + seotbx.utils.const.EPS)
    Alpha = np.sum(pk * alphak, axis=0)
    if full_computation:
        Beta = np.sum(pk * betak, axis=0)
        Delta = np.sum(pk * deltak, axis=0)
        Gamma = np.sum(pk * gammak, axis=0)
        Lambda = np.sum(pk * lambdak, axis=0)
    M_out[defs.Entropy] = Entropy
    M_out[defs.Anisotropy] = Anisotropy
    M_out[defs.Alpha] = Alpha
    if full_computation:
        M_out[defs.Beta] = Beta
        M_out[defs.Delta] = Delta
        M_out[defs.Gamma] = Gamma
        M_out[defs.Lambda] = Lambda
        M_out[defs.Alpha1] = alphak[0]
        M_out[defs.Alpha2] = alphak[1]
        M_out[defs.Alpha3] = alphak[2]
        M_out[defs.Lambda1] = lambdak[0]
        M_out[defs.Lambda2] = lambdak[1]
        M_out[defs.Lambda3] = lambdak[2]
    return M_out.reshape(finale_shape)
