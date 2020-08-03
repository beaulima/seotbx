import logging

logger = logging.getLogger("pyatcortbx.polsarproc.decomposition")
import numpy as np
import scipy
import seotbx
import seotbx.polsarproc.definitions as defs
import matplotlib.pyplot as plt

EPS = 1e-10
RAD2DEG = 180.0 / np.pi
LOG3 = np.log(3)

import seotbx
import seotbx.polsarproc.test_paths as ts
import seotbx.polsarproc.definitions as defs

roi = ts.ROI


def curve1_halpha():
    """
     see. Polarization (p.99)
    :return:
    """
    M_In = []
    for m in np.arange(0, 1.0 + 0.1, 0.01):
        M_In.append([[1, 0, 0], [0, m, 0], [0, 0, m]])
    M_In = np.array(M_In).transpose(1, 2, 0)
    halpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(M_In)
    return halpha

def curve2_halpha():
    """
    see. Polarization (p.99)
    :return:
    """
    M_In = []
    step = 0.01
    for m in np.arange(0, 0.0001 + 0.00001, 0.00001):
        M_In.append([[0, 0, 0], [0, 1, 0], [0, 0, 2*m]])
    for m in np.arange(0.0001, 0.5 + step, step):
        M_In.append([[0, 0, 0], [0, 1, 0], [0, 0, 2*m]])
    for m in np.arange(0.5, 1.0 + step, step):
        M_In.append([[2*m-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    M_In = np.array(M_In).transpose(1, 2, 0)
    halpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(M_In)
    return halpha


def haalpha_plot(M_in, title=None, bshow=True, save_path=None):
    import seaborn as sns
    sns.set()
    import matplotlib.pyplot as plt

    MARKER_SIZE = 1
    curve1 = curve1_halpha()
    curve2 = curve2_halpha()

    plt.scatter(M_in[defs.Entropy], M_in[defs.Alpha], s=MARKER_SIZE)
    plt.xlabel(r"Entropy ($H$)")
    plt.ylabel(r"$\alpha$ [$^{\circ}$]")
    if title is None:
        plt.title(r"$H/\alpha$ diagram")

    plt.plot(curve1[defs.Entropy], curve1[defs.Alpha], 'r')
    plt.plot(curve2[defs.Entropy], curve2[defs.Alpha], 'g')
    plt.xlim(0, 1.0)
    plt.ylim(0, 90.0)
    if bshow:
        plt.show()

    plt.scatter(M_in[defs.Anisotropy], M_in[defs.Alpha], s=MARKER_SIZE)
    plt.xlabel(r"Anisotropy ($A$)")
    plt.ylabel(r"$\alpha$ [$^{\circ}$]")
    if title is None:
        plt.title(r"$A/\alpha$ diagram")
    plt.xlim(0, 1.0)
    plt.ylim(0, 90.0)
    if bshow:
        plt.show()

    plt.scatter(M_in[defs.Entropy], M_in[defs.Anisotropy], s=MARKER_SIZE)
    plt.xlabel(r"Entropy ($H$)")
    plt.ylabel(r"Anisotropy ($A$)")
    if title is None:
        plt.title(r"$H/A$ diagram")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.plot(curve1[defs.Entropy], curve1[defs.Anisotropy], 'r')
    plt.plot(curve2[defs.Entropy], curve2[defs.Anisotropy], 'g')

    if bshow:
        plt.show()


def t3_haalpha_decomposition(M_in, full_computation=True):
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
    pk = lambdak / (np.sum(lambdak, axis=0) + EPS)
    pk[pk > 1.0] = 1.0
    pk[pk < 0.0] = 0.0
    alphak = np.arccos(np.absolute(V0)) * RAD2DEG
    if full_computation:
        betak = np.arctan2(np.absolute(V2), EPS + np.absolute(V1)) * RAD2DEG
        phik = np.arctan2(V0.imag, EPS + V0.real)
        deltak = np.arctan2(V1.imag, EPS + V1.real) - phik
        deltak = np.arctan2(np.sin(deltak), np.cos(deltak) + EPS) * RAD2DEG
        gammak = np.arctan2(V2.imag, EPS + V2.real) - phik
        gammak = np.arctan2(np.sin(gammak), np.cos(gammak) + EPS) * RAD2DEG
    Entropy = -np.sum(pk * np.log(pk + EPS), axis=0) / LOG3
    Entropy[Entropy < 0] = 0.0
    Anisotropy = (pk[1] - pk[2]) / (pk[1] + pk[2] + EPS)
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
