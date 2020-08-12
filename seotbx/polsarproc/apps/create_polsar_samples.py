import logging
import seotbx
import numpy as np
import seotbx.polsarproc.definitions as defs
logger = logging.getLogger("seotbx.polsarproc.apps.create_polsar_samples")
from sklearn.cluster import KMeans

def polsarsamples_parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="create synthetic polarimetric samples from sigs")
    ap.add_argument("sigs_dataset", type=str, help="path to the polarimetric signatures datasets")
    ap.add_argument("save_dir", type=str, help="path to the session output root directory")

def draw_umap(data, c, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    import umap
    import matplotlib.pyplot as plt
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=c, s=0.5)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=c, s=0.5)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=c, s=0.5)
    plt.title(title, fontsize=18)
    plt.show()





def polsarsamples_application_func(args):
    """Synthetic polarimetric signatures.

    """
    signatures_dataset = None
    try:
        signatures_dataset = np.load(args.sigs_dataset)
        logger.info(f"load file {signatures_dataset.shape} ({signatures_dataset.dtype}): {args.sigs_dataset}")
    except:
        logger.fatal(f"unable to load file: {args.sigs_dataset}")

    signatures_haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(
        signatures_dataset.transpose(1, 2, 0), False)

    seotbx.polsarproc.viz.haalpha_plot(M_in=signatures_haalpha, bshow=True)

    #halpha

    M_in0 = signatures_haalpha.transpose(1, 0)
    signatures_class_mean = []
    colors = []
    for key in defs.HALPHA_DIV:

        if key == defs.OMITCLASS:
            continue

        #if key != "Z1":
        #    continue

        info = defs.HALPHA_DIV[key]
        color = info[3]

        h_lim = info[4]
        alpha_lim = info[5]

        if key == "Z3" :
            h_lim=(0.0, 0.25)

        if key == "Z1" or  key == "Z2":
            h_lim=(0.0, 0.12)

        if key == "Z1" or key == "Z4" or key == "Z7" :
            alpha_lim=(68, 70)

        if key == "Z5":
            alpha_lim=(43, 47)

        if key == "Z3":
            alpha_lim = (25, 30)

        if key == "Z6":
            alpha_lim = (30, 35)

        if key == "Z8":
            alpha_lim = (47, 51)

        if key == "Z4" :
            h_lim=(0.60, 0.70)

        if key == "Z4" or key == "Z5" or key == "Z6" :
            h_lim=(0.50, 0.60)

        if key == "Z7" or key == "Z8" or key == "Z9" :
            h_lim=(0.90, 0.92)

        idx_min_h = M_in0[:, defs.Entropy] >= h_lim[0]
        idx_max_h = M_in0[:, defs.Entropy] < h_lim[1]
        idx_min_alpha = M_in0[:, defs.Alpha] >= alpha_lim[0]
        idx_max_alpha = M_in0[:, defs.Alpha] < alpha_lim[1]
        idx = idx_min_h & idx_max_h & idx_min_alpha & idx_max_alpha

        signatures_class = np.expand_dims(np.median(signatures_dataset[idx], axis=(0)), axis=0)

        for k in range(signatures_class.shape[0]):
            signatures_class_mean.append(signatures_class[k])
            colors.append(color)

    signatures_class_mean = np.array(signatures_class_mean)
    signatures_haalpha_class_mean = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(
        signatures_class_mean.transpose(1, 2, 0), False)

    seotbx.polsarproc.viz.haalpha_plot(M_in=signatures_haalpha_class_mean, bshow=True, marker_size=10)

    H=256
    W=256
    n_samples = H*W
    n_looks = 16
    import seaborn as sns
    from scipy import stats
    import matplotlib.pyplot as plt
    for M_in in signatures_class_mean:
        M_out = seotbx.polsarproc.sim.polsar_n_looks_simulation(M_in, n_looks, n_samples)
        X_out = seotbx.polsarproc.convert.MX3_to_X3(np.array(M_out).transpose(1, 2, 0)).reshape(9, H, W)

        enl = (np.mean(X_out[defs.T11])/np.std(X_out[defs.T11]))**2
        sns.distplot(X_out[defs.T11], kde=False, rug=True, fit=stats.gamma)
        alpha, loc, beta = stats.gamma.fit(X_out[defs.T11])
        print(alpha, loc, beta, enl)
        plt.show()
        rgb = seotbx.polsarproc.viz.rgb_pauli_from_t3_polsar(X_out)

        plt.imshow(rgb.transpose(1, 2, 0))
        plt.show()
        exit(1)


    #M_out_haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(
    #        samples_M_nlooks_outs.transpose(1, 2, 0), False)

    #seotbx.polsarproc.viz.haalpha_plot(M_in=M_out_haalpha, bshow=True, marker_size=0.5)
