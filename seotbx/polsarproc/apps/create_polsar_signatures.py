import logging
import seotbx
import numpy as np

logger = logging.getLogger("seotbx.polsarproc.apps.create_polsar_signatures")

def compute_semi_positive_hermitian_matrix(x, spf=0, n=1):
    g = (x.conjugate().T).transpose(2, 0, 1)
    if spf:
        sigs = x[:]*g[:]
    else:
        sigs = 0.5*(x + g)

    i3 = 1*np.eye(3) + 1j*0*np.eye(3)
    sigs += i3
    sigs = np.power(sigs, n)
    span = (sigs[:, 0, 0] + sigs[:, 1, 1] + sigs[:, 2, 2]).real
    sigs = (sigs.transpose(1, 2, 0)/span).transpose(2, 0, 1)
    sigs0 = []
    for sig in sigs:
        try:
            L = np.linalg.cholesky(sig)
            sigs0.append(sig)
        except:
            toto = 0
    sigs0 = np.array(sigs0)
    return sigs0

def get_sigs_from_random(num_samples, minmax=(-10, 10), spf=0, n=1):
    MIN = minmax[0]
    MAX = minmax[1]
    x = np.random.uniform(MIN, MAX, (num_samples, 3, 3)) + 1j * np.random.uniform(MIN, MAX, (3, 3))
    return compute_semi_positive_hermitian_matrix(x, spf=spf, n=n)

def get_sigs_from_poisson(num_samples, p=2, spf=0, n=1):
    P = p
    x = np.random.poisson(P, (num_samples, 3, 3)) + 1j * np.random.poisson(P, (num_samples, 3, 3))
    return compute_semi_positive_hermitian_matrix(x, spf=spf, n=n)


def polsarsigs_parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="create synthetic polarimetric signatures")
    ap.add_argument("cfg_path", type=str, help="path to the session configuration file")
    ap.add_argument("save_dir", type=str, help="path to the session output root directory")


def polsarsigs_application_func(args):
    """Synthetic polarimetric signatures.

    """

    #config = thelper.utils.load_config(args.cfg_path)
    config = None
    save_dir = args.save_dir

    h_resolution = 0.05
    a_resolution = 0.05
    alpha_resolution = 1.0

    h_size = int(1.0 / h_resolution)
    a_size = int(1.0 / a_resolution)
    alpha_size = int(90.0 / alpha_resolution)

    data_count = np.zeros((h_size, a_size, alpha_size)).astype('uint16')

    count_voxels = h_size*a_size*alpha_size
    selected_signatures = []
    halpha_data = []
    idx_data = []
    MAX_COUNTS = 1

    nposs = count_voxels*MAX_COUNTS
    logger.info(f"Number of possiblity: {nposs}")
    NSAMPLES_PER_PRODUCTION = 100000

    spf = 1
    n = 1
    nselections = 0
    MAX_PC = 0.80
    niters = 0
    while nselections/float(nposs) < MAX_PC:
        signatures_datasets = []
        for j in range(50):
            if np.random.randint(0,1):
                func = get_sigs_from_random
            else:
                func = get_sigs_from_poisson
            signatures_datasets.append(func(num_samples=NSAMPLES_PER_PRODUCTION,spf=spf, n=n))

        for j, signatures_dataset in enumerate(signatures_datasets):
            haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(signatures_dataset.transpose(1, 2, 0),
                                                                               False).transpose(1, 0)
            haalpha_idx = (haalpha / np.array([h_resolution, a_resolution, alpha_resolution])).astype('uint8')
            for idxs, sig, haalpha1 in zip(haalpha_idx, signatures_dataset, haalpha):
                if data_count[idxs[0]][idxs[1]][idxs[2]] < MAX_COUNTS:
                    data_count[idxs[0]][idxs[1]][idxs[2]] += 1
                    selected_signatures.append(sig)
                    halpha_data.append(haalpha1)
                    idx_data.append(idxs)

        nselections = len(selected_signatures)
        logger.info(f"{nselections/float(nposs) * 100.0:.2f}%")
        niters += 1

    dtobj = seotbx.utils.get_now()
    signatures_datasetsT = np.array(selected_signatures)
    save_sigs_filepath = seotbx.utils.create_path_with_timestamp(dirpath=save_dir, basename="synthetic_polsar_sigs",
                                                                 ext='npy', dtobj=dtobj)
    with open(save_sigs_filepath, "wb") as f:
        np.save(f, signatures_datasetsT)

    logger.info(f"save polarimetric signatures {signatures_datasetsT.shape}: {save_sigs_filepath}")

    signatures_datasetsT0 = np.load(save_sigs_filepath)

    signatures_haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(
        signatures_datasetsT0.transpose(1, 2, 0), False)

    seotbx.polsarproc.viz.haalpha_plot(M_in=signatures_haalpha, bshow=True, save_dirpath=save_dir, dtobj=dtobj)






