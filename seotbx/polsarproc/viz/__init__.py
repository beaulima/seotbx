import logging
logger = logging.getLogger("pyatcortbx.viz")
import seotbx
import seotbx.polsarproc.definitions as defs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def linear_transform(x0, y0, x1, y1):
    m = (y1 - y0) / float(x1 - x0)
    b = y1 - m * x1
    return m, b


def linear_q_stretching(raster_bands, min_q=0.05, max_q=0.95, min_v=0.0, max_v=1.0):
    min_q = min_q
    max_q = max_q
    min_y = min_v
    max_y = max_v
    raster_bands0 = []
    for k in range(raster_bands.shape[0]):
        min_x = np.quantile(raster_bands[k], min_q)
        max_x = np.quantile(raster_bands[k], max_q)
        m, b = linear_transform(min_x, min_y, max_x, max_y)
        raster_bands0.append(m * raster_bands[k] + b)
    raster_bands0 = np.dstack(raster_bands0).transpose(2, 0, 1)
    np.clip(raster_bands0, min_y, max_y)
    return raster_bands0


def linear_sigma_stretching(raster_bands, sigma=1.5, min_v=0.0, max_v=1.0):
    min_y = min_v
    max_y = max_v
    raster_bands0 = []
    for k in range(raster_bands.shape[2]):
        avg = np.mean(raster_bands[k])
        stdev = np.std(raster_bands[k])
        min_x = avg - sigma * stdev
        max_x = avg + sigma * stdev
        m, b = linear_transform(min_x, min_y, max_x, max_y)
        raster_bands0.append(m * raster_bands[k] + b)
    raster_bands0 = np.dstack(raster_bands0)
    np.clip(raster_bands0, min_y, max_y)
    return raster_bands0


def rgb_pauli_from_s2_polsar(raster_bands, min_q=0.05, max_q=0.95, min_v=0.0, max_v=1.0):
    i_HH = raster_bands[defs.HH].real
    i_VV = raster_bands[defs.VV].real
    i_HV = raster_bands[defs.HV].real
    i_VH = raster_bands[defs.VH].real
    q_HH = raster_bands[defs.HH].imag
    q_VV = raster_bands[defs.VV].imag
    q_HV = raster_bands[defs.HV].imag
    q_VH = raster_bands[defs.VH].imag
    r = ((i_HH - i_VV) * (i_HH - i_VV) + (q_HH - q_VV) * (q_HH - q_VV)) / 2
    g = ((i_HV + i_VH) * (i_HV + i_VH) + (q_HV + q_VH) * (q_HV + q_VH)) / 2
    b = ((i_HV + i_VH) * (i_HV + i_VH) + (q_HV + q_VH) * (q_HV + q_VH)) / 2
    rgb = np.dstack([r, g, b]).transpose(2, 0, 1)
    rgb = linear_q_stretching(rgb, min_q, max_q, min_v, max_v)
    return rgb


def lrgb_pauli_from_s2_polsar(raster_bands, min_vi, max_vi, min_vf=0.0, max_vf=1.0):
    i_HH = raster_bands[defs.HH].real
    i_VV = raster_bands[defs.VV].real
    i_HV = raster_bands[defs.HV].real
    i_VH = raster_bands[defs.VH].real
    q_HH = raster_bands[defs.HH].imag
    q_VV = raster_bands[defs.VV].imag
    q_HV = raster_bands[defs.HV].imag
    q_VH = raster_bands[defs.VH].imag
    r = ((i_HH - i_VV) * (i_HH - i_VV) + (q_HH - q_VV) * (q_HH - q_VV)) / 2
    g = ((i_HV + i_VH) * (i_HV + i_VH) + (q_HV + q_VH) * (q_HV + q_VH)) / 2
    b = ((i_HV + i_VH) * (i_HV + i_VH) + (q_HV + q_VH) * (q_HV + q_VH)) / 2
    rgb = np.dstack([r, g, b])
    rgb = linear_q_stretching(rgb, min_vi, max_vi, min_vf, max_vf)
    return rgb


def rgb_pauli_from_t3_polsar(raster_bands, min_q=0.05, max_q=0.95, min_v=0.0, max_v=1.0):
    rgb = raster_bands[[defs.PAULI_T3_VIZ_IDX["R"], defs.PAULI_T3_VIZ_IDX["G"], defs.PAULI_T3_VIZ_IDX["B"]]]
    rgb = linear_q_stretching(rgb, min_q, max_q, min_v, max_v)
    return rgb


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


def halpha_plot_handle(M_in,
                 bclasscolor=True,
                 marker_size=0.5,
                 alpha=0.5):
    """
    Plot the three plots (HAlpha, HA, AAlpha) from input M_in
    """
    sns.set()
    curve1 = curve1_halpha()
    curve2 = curve2_halpha()
    if bclasscolor:
        M_in0 = M_in.transpose(1, 0)
        for key in defs.HALPHA_CLASS_DEF:
            if key == defs.OMITCLASS:
                continue
            sig_info = defs.HALPHA_CLASS_DEF[key]
            color = sig_info["color"]
            h_lims = sig_info["h_lims"]
            al_lims = sig_info["al_lims"]
            idx_min_h = M_in0[:, defs.Entropy] >= h_lims[0]
            idx_max_h = M_in0[:, defs.Entropy] < h_lims[1]
            idx_min_alpha = M_in0[:, defs.Alpha] >= al_lims[0]
            idx_max_alpha = M_in0[:, defs.Alpha] < al_lims[1]
            idx = idx_min_h & idx_max_h & idx_min_alpha & idx_max_alpha
            plt.scatter(M_in[defs.Entropy][idx], M_in[defs.Alpha][idx], s=marker_size, alpha=alpha, color=color)

    else:
        plt.scatter(M_in[defs.Entropy], M_in[defs.Alpha], s=marker_size,alpha=alpha)

    plt.xlabel(r"Entropy ($H$)")
    plt.ylabel(r"$\alpha$ [$^{\circ}$]")
    plt.title(r"$H/\alpha$ diagram")

    plt.plot([defs.lim_H1,defs.lim_H1], [defs.lim_al_min,defs.lim_al_max], 'k--')
    plt.plot([defs.lim_H2, defs.lim_H2], [defs.lim_al_min,defs.lim_al_max], 'k--')

    plt.plot([defs.lim_H_min, defs.lim_H2], [defs.lim_al3,defs.lim_al3], 'k--')
    plt.plot([defs.lim_H_min, defs.lim_H2], [defs.lim_al4,defs.lim_al4], 'k--')

    plt.plot([defs.lim_H2, defs.lim_H1], [defs.lim_al2, defs.lim_al2], 'k--')
    plt.plot([defs.lim_H2, defs.lim_H1], [defs.lim_al5, defs.lim_al5], 'k--')

    plt.plot([defs.lim_H1, defs.lim_H_max], [defs.lim_al1, defs.lim_al1], 'k--')
    plt.plot([defs.lim_H1, defs.lim_H_max], [defs.lim_al5, defs.lim_al5], 'k--')

    for key in defs.HALPHA_CLASS_DEF:
        sig_info = defs.HALPHA_CLASS_DEF[key]
        id_pos = sig_info["id_pos"]
        id = sig_info["id"]
        plt.text(id_pos[0], id_pos[1], rf"${id}$")

    plt.plot(curve1[defs.Entropy], curve1[defs.Alpha], 'r')
    plt.plot(curve2[defs.Entropy], curve2[defs.Alpha], 'g')
    plt.xlim(defs.lim_H_min, defs.lim_H_max)
    plt.ylim(defs.lim_al_min, defs.lim_al_max)


def haalpha_plot(M_in, bshow: bool=True, save_dirpath: str = "", dtobj=None,
                 bclasscolor=True,
                 marker_size=0.5,
                 alpha=0.5,
                 suffix=""):
    """
    Plot the three plots (HAlpha, HA, AAlpha) from input M_in
    """
    sns.set()

    curve1 = curve1_halpha()
    curve2 = curve2_halpha()

    plt.clf()
    halpha_plot_handle(M_in=M_in,
                 bclasscolor=bclasscolor,
                 marker_size=marker_size,
                 alpha=alpha)
    if dtobj is None:
        dtobj = seotbx.utils.get_now()

    if save_dirpath != "":
        basename ="HALPHA"
        if suffix != "":
            basename = f"{basename}_{suffix}"
        fig_name = seotbx.utils.create_path_with_timestamp(dirpath=save_dirpath,
                                                           basename=basename,
                                                           ext="png",
                                                           dtobj=dtobj)
        plt.savefig(fig_name)
    if bshow:
        plt.show()

    plt.clf()
    if bclasscolor:
        M_in0 = M_in.transpose(1, 0)
        for key in defs.HALPHA_CLASS_DEF:
            if key == defs.OMITCLASS:
                continue
            sig_info = defs.HALPHA_CLASS_DEF[key]
            color = sig_info["color"]
            h_lims = sig_info["h_lims"]
            al_lims = sig_info["al_lims"]
            idx_min_h = M_in0[:, defs.Entropy] >= h_lims[0]
            idx_max_h = M_in0[:, defs.Entropy] < h_lims[1]
            idx_min_alpha = M_in0[:, defs.Alpha] >= al_lims[0]
            idx_max_alpha = M_in0[:, defs.Alpha] < al_lims[1]
            idx = idx_min_h & idx_max_h & idx_min_alpha & idx_max_alpha
            plt.scatter(M_in[defs.Anisotropy][idx], M_in[defs.Alpha][idx], s=marker_size, alpha=alpha, color=color)

    else:
        plt.scatter(M_in[defs.Anisotropy], M_in[defs.Alpha], s=marker_size,alpha=alpha)
    plt.xlabel(r"Anisotropy ($A$)")
    plt.ylabel(r"$\alpha$ [$^{\circ}$]")
    plt.title(r"$A/\alpha$ diagram")
    plt.xlim(defs.lim_A_min, defs.lim_A_max)
    plt.ylim(defs.lim_al_min, defs.lim_al_max)
    if save_dirpath != "":
        basename = "AALPHA"
        if suffix != "":
            basename = f"{basename}_{suffix}"
        fig_name = seotbx.utils.create_path_with_timestamp(dirpath=save_dirpath,
                                                           basename=basename,
                                                           ext="png",
                                                           dtobj=dtobj)
        plt.savefig(fig_name)
    if bshow:
        plt.show()

    plt.clf()
    if bclasscolor:
        M_in0 = M_in.transpose(1, 0)
        for key in defs.HALPHA_CLASS_DEF:
            if key == defs.OMITCLASS:
                continue
            sig_info = defs.HALPHA_CLASS_DEF[key]
            color = sig_info["color"]
            h_lims = sig_info["h_lims"]
            al_lims = sig_info["al_lims"]
            idx_min_h = M_in0[:, defs.Entropy] >= h_lims[0]
            idx_max_h = M_in0[:, defs.Entropy] < h_lims[1]
            idx_min_alpha = M_in0[:, defs.Alpha] >= al_lims[0]
            idx_max_alpha = M_in0[:, defs.Alpha] < al_lims[1]
            idx = idx_min_h & idx_max_h & idx_min_alpha & idx_max_alpha
            plt.scatter(M_in[defs.Entropy][idx], M_in[defs.Anisotropy][idx], s=marker_size, alpha=alpha, color=color)

    else:
        plt.scatter(M_in[defs.Anisotropy], M_in[defs.Alpha], s=marker_size, alpha=alpha)

    plt.xlabel(r"Entropy ($H$)")
    plt.ylabel(r"$Anisotropy$]")
    plt.title(r"$H/A$ diagram")
    plt.xlim(defs.lim_H_min, defs.lim_H_max)
    plt.ylim(defs.lim_A_min, defs.lim_A_max)
    if save_dirpath != "":
        basename = "HA"
        if suffix != "":
            basename = f"{basename}_{suffix}"
        fig_name = seotbx.utils.create_path_with_timestamp(dirpath=save_dirpath,
                                                           basename=basename,
                                                           ext="png",
                                                           dtobj=dtobj)
        plt.savefig(fig_name)
    if bshow:
        plt.show()