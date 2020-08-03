import logging
import seotbx
import numpy as np

logger = logging.getLogger("seotbx.polsarproc.apps.create_polsar_signatures")


def polsarsigs_parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="create synthetic polarimetric signatures")
    ap.add_argument("cfg_path", type=str, help="path to the session configuration file")
    ap.add_argument("save_dir", type=str, help="path to the session output root directory")


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

# def get_cov3_mat(Shh, Shv, Svv):
#
#     shh = (Shh * Shh.conjugate()).real
#     shv = (Shv * Shv.conjugate()).real
#     svv = (Svv * Svv.conjugate()).real
#     rho1 = Shh * Svv.conjugate() / np.sqrt(shh * svv)
#     rho2 = Shh * Shv.conjugate() / np.sqrt(shh * shv)
#     rho3 = Shv * Svv.conjugate() / np.sqrt(shv * svv)
#     ep = shv / shh
#     gm = svv / shh
#     A=1.0
#     B=np.sqrt(2*ep)*rho2
#     C= rho1*np.sqrt(gm)
#     D=2*ep
#     E=np.sqrt(2*gm*ep)*rho3
#     F=gm
#     cov = shh* np.array([[A,B,C],
#               [B.conjugate(),D,E],
#               [C.conjugate(),E.conjugate(),F]], dtype=complex)
#     L = np.linalg.cholesky(cov)
#     return cov


def get_sigs_from_random(num_samples, minmax=(-10, 10), spf=0, n=1):
    MIN = minmax[0]
    MAX = minmax[1]
    x = np.random.uniform(MIN, MAX, (num_samples, 3, 3)) + 1j * np.random.uniform(MIN, MAX, (3, 3))

    Shh = np.random.uniform(-1,1)+ 1j*np.random.uniform(-1,1)
    Shv = np.random.uniform(-1,1)+ 1j*np.random.uniform(-1,1)
    Svv = np.random.uniform(-1,1)+ 1j*np.random.uniform(-1,1)
    #cov = get_cov3_mat(Shh, Shv, Svv)
    return compute_semi_positive_hermitian_matrix(x, spf=spf, n=n)


def get_sigs_from_poisson(num_samples, p=2, spf=0, n=1):
    P = p
    x = np.random.poisson(P, (num_samples, 3, 3)) + 1j * np.random.poisson(P, (num_samples, 3, 3))
    return compute_semi_positive_hermitian_matrix(x, spf=spf, n=n)




def polsarsigs_application_func(args):
    """Synthetic polarimetric signatures.

    """

    #config = thelper.utils.load_config(args.cfg_path)
    config = None
    save_dir = args.save_dir

    h_step = 0.05
    a_step = 0.05
    alpha_step = 1.0

    h_size = int(1.0 / h_step)
    a_size = int(1.0 / a_step)
    alpha_size = int(90.0 / alpha_step)

    data_count = np.zeros((h_size, a_size, alpha_size)).astype('uint16')
    count_voxels = h_size*a_size*alpha_size
    selected_signatures = []
    halpha_data = []
    idx_data = []
    MAX_COUNTS = 1

    NSAMPLES_PER_PRODUCTION = 1000000

    spf = 1
    n = 1
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
        haalpha_idx = (haalpha / np.array([h_step, a_step, alpha_step])).astype('uint8')
        for idxs, sig, haalpha1 in zip(haalpha_idx, signatures_dataset, haalpha):
            if data_count[idxs[0]][idxs[1]][idxs[2]] < MAX_COUNTS:
                data_count[idxs[0]][idxs[1]][idxs[2]] += 1
                selected_signatures.append(sig)
                halpha_data.append(haalpha1)
                idx_data.append(idxs)
        print(j, len(selected_signatures))


    #print(data_count)
    #print(np.count_nonzero(data_count))
    signatures_datasets0 = np.array(selected_signatures)
    signatures_haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(signatures_datasets0.transpose(1,2,0), False)
    seotbx.polsarproc.decomposition.haalpha_plot(signatures_haalpha)
    halpha_data0 = np.array(halpha_data)
    seotbx.polsarproc.decomposition.haalpha_plot(halpha_data0.transpose(1,0))
    #for k in range(len(idx_data)):
    #    idx = idx_data[k]
    #    h = halpha_data[k]
    #    print(h[0], idx[0], h[2], idx[2])