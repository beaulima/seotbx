import logging
import seotbx
import numpy as np
import seotbx.polsarproc.definitions as defs
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import copy
logger = logging.getLogger("seotbx.polsarproc.apps.create_polsar_samples")

def save2json(filepath, obj):

    import json
    obj0 = copy.deepcopy(obj)
    for key in defs.HALPHA_CLASS_DEF:
        if key is defs.OMITCLASS:
            continue
        if "M" in obj0[key]:
            obj0[key]["M"] = seotbx.polsarproc.convert.MX3_to_X3(obj0[key]["M"]).tolist()
        if "M_samples" in obj0[key]:
            obj0[key]["M_samples"] = seotbx.polsarproc.convert.MX3_to_X3(obj0[key]["M_samples"].transpose(1,2,0)).tolist()
    with open(filepath, "w") as fp:
        json.dump(obj0, fp,  indent=4)

def polsarsamples_parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="create (HxW) synthetic polarimetric samples from sigs")
    ap.add_argument("sigs_dataset", type=str, help="path to the polarimetric signatures datasets")
    ap.add_argument("save_dir", type=str, help="path to the session output root directory")
    ap.add_argument("-H", "--height", default=256, type=int, help="sample height")
    ap.add_argument("-W", "--width", default=256, type=int, help="sample width")
    ap.add_argument("-N", "--n_looks", default=1, type=int, help="number of looks")


def plot_intensity_histogram(X_out, sig_info, band_id, n_looks, xlim=(0, 3.0)):
    data = X_out[band_id]
    enl = (np.mean(data) / np.std(data)) ** 2
    sns.distplot(data, kde=False, rug=True, fit=stats.gamma)
    alpha, loc, beta = stats.gamma.fit(data)
    plt.annotate(xycoords='figure fraction', xy=(0.7, 0.75), text=rf"ENL = {enl:.2f}")
    plt.annotate(xycoords='figure fraction', xy=(0.7, 0.70), text=rf"ALPHA = {alpha:.2f}")
    plt.annotate(xycoords='figure fraction', xy=(0.7, 0.65), text=rf"LOC = {loc:.2f}")
    plt.annotate(xycoords='figure fraction', xy=(0.7, 0.60), text=rf"BETA = {beta:.2f}")
    plt.title(rf"Homogeneous simulation {n_looks} look(s)")
    plt.ylabel(r"Distribution (%)")
    plt.xlabel(r"Intensity")
    plt.xlim(xlim[0], xlim[1])
    en_id = sig_info["en_id"]
    id = sig_info["id"]
    plt.suptitle(rf"{defs.T3_IDX_NAME[band_id]} - {en_id} (id)")
    basename = f"{en_id}_{id}_{defs.T3_IDX_NAME[band_id]}_nlooks={n_looks}_histogram".replace(" ", "_")
    return {"enl": enl, "alpha": alpha, "loc": loc, "beta": beta, "basename": basename}


def plot_halpha_with_centers(M_in_haalpha, sig_info, n_looks, halpha_center):
    seotbx.polsarproc.viz.halpha_plot_handle(M_in=M_in_haalpha, marker_size=0.5)
    en_id = sig_info["en_id"]
    id = sig_info["id"]
    plt.suptitle(rf"{en_id} ({id}) - {n_looks} look(s)")
    plt.scatter(halpha_center[defs.Entropy], halpha_center[defs.Alpha],
                s=20, facecolor="None", alpha=1.0, color='k')
    plt.annotate(xycoords='figure fraction', xy=(0.69, 0.3),
                 text=rf"H = {halpha_center[defs.Entropy]:.2f}")
    plt.annotate(xycoords='figure fraction', xy=(0.69, 0.25),
                 text=rf"A = {halpha_center[defs.Anisotropy]:.2f}")
    plt.annotate(xycoords='figure fraction', xy=(0.69, 0.20),
                 text=r"$\bar{\alpha}$" + rf" = {halpha_center[defs.Alpha]:.2f}" + r"$^{\circ}$")
    basename = f"{en_id}_{id}_nlooks={n_looks}_halpha_plot"
    basename = basename.replace(" ", "_")
    return {"H": halpha_center[defs.Entropy], "A": halpha_center[defs.Anisotropy], "alpha": halpha_center[defs.Alpha],
            "basename": basename}


def produce_signatures_sample_report(sigs_samples_info, save_dir, dtobj, bsave, bshow):
    for key in defs.HALPHA_CLASS_DEF:
        # bypass Z9
        if key == defs.OMITCLASS:
            continue
        sig_info = sigs_samples_info[key]["info"]
        M_samples = sigs_samples_info[key]["M_samples"]
        M = sigs_samples_info[key]["M"]
        n_looks = sigs_samples_info[key]["n_looks"]
        height = sigs_samples_info[key]["height"]
        width = sigs_samples_info[key]["width"]

        M_haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(
            M, False)

        X_out = seotbx.polsarproc.convert.MX3_to_X3(np.array(M_samples).transpose(1, 2, 0)).reshape(9, height, width)
        for band_id in [defs.T11, defs.T22, defs.T33]:
            plt.clf()
            info = plot_intensity_histogram(X_out, sig_info, band_id, n_looks)
            if bsave:
                save_fig(basename=info["basename"], save_dir=save_dir, dtobj=dtobj)
            if bshow:
                plt.show()

        M_out_haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(M_samples.transpose(1, 2, 0), False)
        seotbx.polsarproc.viz.halpha_plot_handle(M_in=M_out_haalpha, marker_size=0.5)
        halpha_center = (M_haalpha[defs.Entropy],
                         M_haalpha[defs.Anisotropy],
                         M_haalpha[defs.Alpha])
        plt.clf()
        info = plot_halpha_with_centers(M_out_haalpha, sig_info, n_looks, halpha_center)
        if bsave:
            basename = info["basename"]
            basename = basename.replace(" ", "_")
            save_fig(basename=basename, save_dir=save_dir, dtobj=dtobj)

        if bshow:
            plt.show()


def save_fig(basename, save_dir, dtobj, ext="png"):
    basename = basename.replace(" ", "_")
    filename = seotbx.utils.create_path_with_timestamp(dirpath=save_dir,
                                                       basename=basename,
                                                       ext=ext,
                                                       dtobj=dtobj)
    plt.savefig(filename)


def modify_lims(key, h_lim, alpha_lim):
    if key == "Z3":
        h_lim = (0.0, 0.25)

    if key == "Z1" or key == "Z2":
        h_lim = (0.0, 0.12)

    if key == "Z1" or key == "Z4" or key == "Z7":
        alpha_lim = (68, 70)

    if key == "Z5":
        alpha_lim = (43, 47)

    if key == "Z3":
        alpha_lim = (25, 30)

    if key == "Z6":
        alpha_lim = (30, 35)

    if key == "Z8":
        alpha_lim = (47, 51)

    if key == "Z4":
        h_lim = (0.60, 0.70)

    if key == "Z4" or key == "Z5" or key == "Z6":
        h_lim = (0.50, 0.60)

    if key == "Z7" or key == "Z8" or key == "Z9":
        h_lim = (0.90, 0.92)
    return h_lim, alpha_lim


def polsarsamples_application_func(args):
    """Synthetic polarimetric signatures.

    """
    dtobj = seotbx.utils.get_now()
    signatures_dataset = None
    try:
        signatures_dataset = np.load(args.sigs_dataset)
        logger.info(f"load file {signatures_dataset.shape} ({signatures_dataset.dtype}): {args.sigs_dataset}")
    except:
        logger.fatal(f"unable to load file: {args.sigs_dataset}")

    signatures_haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(
        signatures_dataset.transpose(1, 2, 0), False)

    height = args.height
    width = args.width
    n_samples = height * width
    n_looks = args.n_looks
    M_in0 = signatures_haalpha.transpose(1, 0)
    signatures_class_mean = []
    save_dir = args.save_dir
    bsave = False
    bshow = False
    if save_dir != "":
        bsave = True

    seotbx.polsarproc.viz.haalpha_plot(M_in=signatures_haalpha, bshow=False, save_dirpath=save_dir, dtobj=dtobj,
                                       suffix="all")

    colors = []
    for key in defs.HALPHA_CLASS_DEF:

        if key == defs.OMITCLASS:
            continue

        # if key != "Z1":
        #    continue

        sig_info = defs.HALPHA_CLASS_DEF[key]
        color = sig_info["color"]

        h_lims = sig_info["h_lims"]
        al_lims = sig_info["al_lims"]

        # This is to assure that centers computed are in their specific class.
        # If the full region is used, some average/median signatures are translated in an another class
        h_lims, al_lims = modify_lims(key, h_lims, al_lims)

        idx_min_h = M_in0[:, defs.Entropy] >= h_lims[0]
        idx_max_h = M_in0[:, defs.Entropy] < h_lims[1]
        idx_min_alpha = M_in0[:, defs.Alpha] >= al_lims[0]
        idx_max_alpha = M_in0[:, defs.Alpha] < al_lims[1]
        idx = idx_min_h & idx_max_h & idx_min_alpha & idx_max_alpha

        signatures_class = np.expand_dims(np.median(signatures_dataset[idx], axis=(0)), axis=0)

        for sig in signatures_class:
            signatures_class_mean.append(seotbx.polsarproc.tools.span_normalize_M3(sig))
            colors.append(color)

    signatures_class_mean = np.array(signatures_class_mean)
    signatures_haalpha_class_mean = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(
        signatures_class_mean.transpose(1, 2, 0), False)

    seotbx.polsarproc.viz.haalpha_plot(M_in=signatures_haalpha_class_mean, bshow=False, marker_size=10,
                                       save_dirpath=save_dir, dtobj=dtobj, suffix="centers")

    sigs_info = {}
    for k, M_in in enumerate(signatures_class_mean):
        sig_info = defs.HALPHA_CLASS_DEF[defs.HALPHA_DIV_IDX[k]]
        sigs_info[defs.HALPHA_DIV_IDX[k]] = {"info": sig_info, "M": M_in}

    if bsave:
        basename = "synthetic_polsar_sigs_base"
        file_path = seotbx.utils.create_path_with_timestamp(dirpath=save_dir,
                                                            basename=basename,
                                                            ext='pkl',
                                                            dtobj=dtobj)
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(sigs_info, f)

        file_path = seotbx.utils.create_path_with_timestamp(dirpath=save_dir,
                                                            basename=basename,
                                                            ext='json',
                                                            dtobj=dtobj)

        save2json(file_path,sigs_info)

    sigs_samples_info = sigs_info
    for key in defs.HALPHA_CLASS_DEF:
        # bypass Z9
        if key == defs.OMITCLASS:
            continue
        M_in = sigs_info[key]["M"]
        M_out = seotbx.polsarproc.tools.polsar_n_looks_simulation(M_in=M_in, n_looks=n_looks, n_samples=n_samples)
        sigs_samples_info[key]["M_samples"] = M_out
        sigs_samples_info[key]["n_looks"] = n_looks
        sigs_samples_info[key]["height"] = height
        sigs_samples_info[key]["width"] = width

    if bsave:
        basename = f"synthetic_polsar_sigs_samples_nlooks={n_looks}"
        file_path = seotbx.utils.create_path_with_timestamp(dirpath=save_dir,
                                                            basename=basename,
                                                            ext='pkl',
                                                            dtobj=dtobj)
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(sigs_samples_info, f)

        file_path = seotbx.utils.create_path_with_timestamp(dirpath=save_dir,
                                                            basename=basename,
                                                            ext='json',
                                                            dtobj=dtobj)

        save2json(file_path, sigs_samples_info)

    if bsave:
        produce_signatures_sample_report(sigs_samples_info, save_dir, dtobj, bsave, bshow)
