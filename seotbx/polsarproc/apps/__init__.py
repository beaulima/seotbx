import logging
import os
import seotbx
import seotbx.polsarproc.definitions as defs
import numpy as np

logger = logging.getLogger(__name__)

def create_polsar_signatures_subparser(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="create synthetic polarimetric signatures")
    ap.add_argument("cfg_path", type=str, help="path to the session configuration file")
    ap.add_argument("save_dir", type=str, help="path to the session output root directory")


def compute_semi_positive_hermitian_matrix(x):
    np.random.seed(0)
    if 0:
        t11 = np.random.uniform(1.0,10) + 1j*0
        t12 = np.random.uniform(0,10) + 1j * np.random.uniform(-10,10)
        t13 = np.random.uniform(-10, 10) + 1j * np.random.uniform(-10, 10)
        t22 = np.random.uniform(0.1, 10) + 1j * 0
        t21 = np.random.uniform(-10, 10) + 1j * np.random.uniform(-10, 10)
        t33 = np.random.uniform(0.1, 10) + 1j * 0

        sigs = np.array([[t11,            t12, t13],
             [t12.conjugate(),t22, t21 ],
             [t13.conjugate(), t21.conjugate(),   t33]])
        sigs /= (sigs[0, 0] + sigs[1, 1] + sigs[2, 2]).real
        haalpha0 = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(sigs)

        MC0 = seotbx.polsarproc.convert.X3_to_MX3(seotbx.polsarproc.definitions.C9_SIG)
        haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(MC0)

    g = (x.conjugate().T).transpose(2, 0, 1)

    sigs = ((0.5 * (x + g))**2).transpose(1, 2, 0)
    sigs /= (sigs[0, 0] + sigs[1, 1] + sigs[2, 2]).real
    sigs0 = sigs.transpose(2,0,1)
    rk = np.linalg.det(sigs.transpose(2,0,1))
    rk0 = np.linalg.det(sigs0[0])
    haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(sigs)
    halpha = haalpha.transpose(1, 0)
    idxs = halpha[:,1] < 0.8
    sigs = sigs[:,:,idxs]
    pp = sigs.transpose(2,0,1)
    ppp = pp[0]
    haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(ppp)
    halpha = haalpha
    L = np.linalg.cholesky(pp)

    return sigs


def get_sigs_from_random(num_samples, minmax=(-1, 1)):
    MIN = minmax[0]
    MAX = minmax[1]
    x = np.random.uniform(MIN, MAX, (num_samples, 3, 3)) + 1j * np.random.uniform(MIN, MAX, (3, 3))
    return compute_semi_positive_hermitian_matrix(x)


def get_sigs_from_poisson(num_samples, p=10):
    P = p
    x = np.random.poisson(P, (num_samples, 3, 3)) + 1j * np.random.poisson(P, (num_samples, 3, 3))
    return compute_semi_positive_hermitian_matrix(x)


def create_polsar_signatures(args):
    """Creates a session to train a model.

    All generated outputs (model checkpoints and logs) will be saved in a directory named after the
    session (the name itself is specified in ``config``), and located in ``save_dir``.

    Args:
        args: input arguments must contain:
            - args.cfg_path, the path to the config dictionary that provides all required data configuration and
            trainer parameters; see:class:`thelper.train.base.Trainer` and :func:`thelper.data.utils.create_loaders`
            for more information. Here, it is only expected to contain a ``name`` field that specifies the name of
            the session.
            - args.save_dir, the path to the root directory where the session directory should be saved. Note that
            this is not the path to the session directory itself, but its parent, which may also contain
            other session directories.

    .. seealso::
        | :class:`thelper.train.base.Trainer`
    """

    config = thelper.utils.load_config(args.cfg_path)
    save_dir = args.save_dir

    session_name = thelper.utils.get_config_session_name(config)
    assert session_name is not None, "config missing 'name' field required for output directory"
    #thelper.logger.info("creating new polsar signatures creation session '%s'..." % session_name)

    h_step = 0.01
    a_step = 0.01
    alpha_step = 90

    h_size = int(1.0 / h_step)
    a_size = int(1.0 / a_step)
    alpha_size = int(90.0 / alpha_step)

    data_count = np.zeros((h_size, a_size, alpha_size)).astype('uint16')

    selected_signatures = []
    halpha_data = []
    MAX_COUNTS = 1

    NSAMPLES_PER_PRODUCTION = 1000000
    for j in range(1):
        signatures_datasets = []
        signatures_datasets.append(get_sigs_from_random(num_samples=NSAMPLES_PER_PRODUCTION))
        for signatures_dataset in signatures_datasets:
            haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(signatures_dataset, False).transpose(
                1, 0)
            haalpha_idx = (haalpha / np.array([h_step, a_step, alpha_step])).astype('uint8')
            for idxs, sig, haalpha1 in zip(haalpha_idx, signatures_dataset.transpose(2, 0, 1), haalpha):
                if data_count[idxs[0]][idxs[1]][idxs[2]] < MAX_COUNTS:
                    data_count[idxs[0]][idxs[1]][idxs[2]] += 1
                    selected_signatures.append(sig)
                    halpha_data.append(haalpha1)
            print(j, len(selected_signatures))

    for j in range(0):
        signatures_datasets = []
        for k in range(1, 10):
            signatures_datasets.append(get_sigs_from_poisson(num_samples=NSAMPLES_PER_PRODUCTION, p=k * 10))
        for signatures_dataset in signatures_datasets:
            haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(signatures_dataset, False).transpose(
                1, 0)
            haalpha_idx = (haalpha / np.array([h_step, a_step, alpha_step])).astype('uint8')
            for idxs, sig, haalpha1 in zip(haalpha_idx, signatures_dataset.transpose(2, 0, 1), haalpha):
                if data_count[idxs[0]][idxs[1]][idxs[2]] < MAX_COUNTS:
                    data_count[idxs[0]][idxs[1]][idxs[2]] += 1
                    # haalpha_test = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(sig, False)
                    selected_signatures.append(sig)
                    halpha_data.append(haalpha1)
            print(j, len(selected_signatures))

    signatures_datasets0 = np.array(selected_signatures).transpose(1, 2, 0)
    signatures_haalpha = seotbx.polsarproc.decomposition.t3_haalpha_decomposition(signatures_datasets0, False)
    seotbx.polsarproc.decomposition.haalpha_plot(signatures_haalpha)


def registering_application():
    from seotbx.apps import registering_application as ra
    ra("polsarsigs", create_polsar_signatures_subparser, create_polsar_signatures)